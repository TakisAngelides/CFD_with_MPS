# Prototype: Shakti's Picard-linearised head equation  M h = rhs  solved in quantics MPS form.
#
#   julia --project=~/CFD_with_MPS mps_headsolve.jl FILE.h5 DAY [ordering] [op_eps] [maxdim] [nsweeps] [x0]
#
# M is the exact sparse matrix Shakti assembled at a saved snapshot (5-point stencil, face
# transmissivities K/dx^2 incl. the Newton term of the creep closure on the diagonal, Dirichlet rows
# at the outlet, frozen rows on the OTHER_BASIN rim). It is written as an MPO
#     A = diag(cP) + diag(cE) Sx+ + diag(cW) Sx- + diag(cN) Sy+ + diag(cS) Sy-
# with the five stencil-coefficient fields (read off M) as MPS, then SVD-compressed.
# Solve with ITensorMPS.linsolve + exact (dense LU) local solves, compare with the sparse Cholesky
# solution, and report MPO ranks, accuracy, time and the conditioning of M.
# Optional: if FILE has 2^n with n < target (env UPSAMPLE=n_target), the fields are bilinearly
# upsampled and the matrix rebuilt from K_x, K_y, gd (synthetic larger grids).

using HDF5, Printf, LinearAlgebra, SparseArrays, Random
include(joinpath(@__DIR__, "..", "..", "Glaciology_SIA", "quantics.jl"))   # read-only use
include(joinpath(@__DIR__, "qtt_tools.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))

const FILE = ARGS[1]
const DAY = parse(Float64, ARGS[2])
const ORD = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :interleaved
const OP_EPS = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-12      # relative truncation of coefficient fields / MPO
const MAXDIM = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 200
const NSWEEPS = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 12
const X0 = length(ARGS) >= 7 ? ARGS[7] : "cold"        # cold | warm (warm = true head truncated to 1e-2)

# --- exact local solver for linsolve (copied from Glaciology_SIA/sia.jl: direct_updater) ---------
function direct_updater(problem, init; internal_kwargs, coefficients, maxdirect = 8000, krylov_kwargs = (;), kwargs...)
    P = problem.reduced_operator
    b = dag(only([contract(c) for c in problem.reduced_constant_terms]))
    a0, a1 = coefficients
    if dim(init) > maxdirect
        x, info = linsolve(P, b, init, a0, a1; krylov_kwargs...)
        return x, (; info)
    end
    op = foldl(*, P.H[(P.lpos + 1):(P.rpos - 1)])
    P.lpos >= 1 && (op = P.LR[P.lpos] * op)
    P.rpos <= length(P.H) && (op = op * P.LR[P.rpos])
    C = combiner(inds(init)...)
    c = combinedind(C)
    Mloc = matrix(op * C * prime(C), prime(c), c)
    a1 != 1 && (Mloc .*= a1)
    a0 != 0 && (Mloc += a0 * LinearAlgebra.I)
    rhsl = vector(b * C)
    F = lu(Mloc; check = false)
    xv = issuccess(F) ? F \ rhsl : qr(Mloc, ColumnNorm()) \ rhsl
    return itensor(xv, c) * dag(C), (; info = nothing)
end

# --- load the snapshot ---------------------------------------------------------------------------
key = @sprintf("day_%05.1f", DAY)
f = h5open(FILE, "r")
g5 = f[key]
nx = length(read(f["x"]))
M = sparse(read(g5["M_I"]), read(g5["M_J"]), read(g5["M_V"]), nx^2, nx^2)
rhs = read(g5["rhs"]); hsp = read(g5["h_sparse"])
t_chol = read(g5["t_chol"]); t_solve = read(g5["t_solve"])
close(f)
n = round(Int, log2(nx))
@printf("snapshot %s %s: %d x %d (n = %d), nnz(M) = %d, ordering %s, op_eps %.0e, maxdim %d\n", FILE, key, nx, nx, n, nnz(M), ORD, OP_EPS, MAXDIM)
@printf("Shakti sparse Cholesky (CHOLMOD, 16 threads): factorise %.4f s + solve %.4f s\n", t_chol, t_solve)

# --- stencil coefficient fields read off M -------------------------------------------------------
lin(i, j) = i + (j - 1) * nx
cP = zeros(nx, nx); cE = zeros(nx, nx); cW = zeros(nx, nx); cN = zeros(nx, nx); cS = zeros(nx, nx)
for j in 1:nx, i in 1:nx
    r = lin(i, j)
    cP[i, j] = M[r, r]
    i < nx && (cE[i, j] = M[r, lin(i + 1, j)])
    i > 1 && (cW[i, j] = M[r, lin(i - 1, j)])
    j < nx && (cN[i, j] = M[r, lin(i, j + 1)])
    j > 1 && (cS[i, j] = M[r, lin(i, j - 1)])
end
# M should be exactly the 5-point stencil
let v = randn(nx^2), V = reshape(v, nx, nx)
    Av = cP .* V
    Av[1:end-1, :] .+= cE[1:end-1, :] .* V[2:end, :]
    Av[2:end, :] .+= cW[2:end, :] .* V[1:end-1, :]
    Av[:, 1:end-1] .+= cN[:, 1:end-1] .* V[:, 2:end]
    Av[:, 2:end] .+= cS[:, 2:end] .* V[:, 1:end-1]
    @printf("stencil check |M v - stencil(v)| / |M v| = %.2e\n", norm(vec(Av) - M * v) / norm(M * v))
end

# --- conditioning of the physical (GROUNDED) block of M: power iteration for lambda_max, inverse
#     iteration with its Cholesky factor for lambda_min (the Dirichlet/frozen rim rows are identity
#     rows and only add eigenvalues 1). Also the explicit stability limit of the parabolic variant
#     e_v dh/dt = div(K grad h) + ...  (forward Euler: dt <= 2 e_v / lambda_max). -----------------
mask = h5open(f -> read(f["mask"]), FILE, "r")
gidx = findall(vec(mask) .== 0)
Mg = M[gidx, gidx]
Fg = cholesky(Symmetric(Mg))
let x = normalize(randn(length(gidx))), lmax = 0.0, lmin = 0.0
    for _ in 1:300; y = Mg * x; lmax = norm(y); x = y / lmax; end
    x = normalize(randn(length(gidx)))
    for _ in 1:80; y = Fg \ x; lmin = 1 / norm(y); x = y * lmin; end
    kf = abs.(filter(!iszero, vcat(vec(cE), vec(cN))))
    @printf("grounded block: cond ~ %.3e (lambda_max %.3e /s, lambda_min %.3e /s); face K/dx^2 range %.2e..%.2e (ratio %.2e)\n",
            lmax / lmin, lmax, lmin, minimum(kf), maximum(kf), maximum(kf) / minimum(kf))
    for ev in (1e-4, 1e-3)
        @printf("  explicit parabolic step limit (e_v = %.0e): dt <= %.3e s  -> %.2e steps per hour\n", ev, 2ev / lmax, 3600 / (2ev / lmax))
    end
end
@printf("sparse residual check: |M h_sparse - rhs|/|rhs| = %.2e\n", norm(M * hsp - rhs) / norm(rhs))

# --- MPS / MPO -----------------------------------------------------------------------------------
g = QuanticsGrid(n, ORD, (0.0, 1.0), (0.0, 1.0))
tomps(A; cutoff = OP_EPS^2) = MPS(to_tensor(A, ORD), g.sites; cutoff = cutoff)
fromtps(m) = from_tensor(Array(contract(m), g.sites...), n, ORD)

# point check of the conversion against idx_to_bits/MPSEvaluator (array_to_mps has been suspect)
let A = randn(nx, nx), m = tomps(A; cutoff = 0.0), E = MPSEvaluator(m)
    err = maximum(abs(E(idx_to_bits(g, (i - 1, j - 1))) - A[i, j]) for (i, j) in zip(rand(1:nx, 50), rand(1:nx, 50)))
    @printf("conversion point check: %.2e\n", err)
end

t_op = @elapsed begin
    fields = (cP, cE, cW, cN, cS)
    mps_c = [tomps(c) for c in fields]
    Sx_p = shift_mpo(g, 1, +1); Sx_m = shift_mpo(g, 1, -1); Sy_p = shift_mpo(g, 2, +1); Sy_m = shift_mpo(g, 2, -1)
    terms = MPO[diag_mpo(mps_c[1])]
    for (c, S) in zip(mps_c[2:end], (Sx_p, Sx_m, Sy_p, Sy_m))
        push!(terms, apply(diag_mpo(c), S; alg = "naive", truncate = false))
    end
    A = truncate!(exact_sum(terms...); cutoff = OP_EPS^2)
end
@printf("coefficient-field ranks (cP cE cW cN cS): %s\n", join(maxlinkdim.(mps_c), " "))
@printf("MPO: max rank %d (bond dims %s), %d params (sparse M: %d nnz); built in %.1f s\n", maxlinkdim(A),
        join(linkdims(A), ","), nparams(A), nnz(M), t_op)

b = tomps(reshape(rhs, nx, nx); cutoff = 1e-28)
href = reshape(hsp, nx, nx)
hmps_exact = tomps(href; cutoff = 1e-28)
@printf("rank(rhs) = %d, rank(h exact, cutoff 1e-14 rel) = %d\n", maxlinkdim(b), maxlinkdim(hmps_exact))
for e in (1e-2, 1e-4, 1e-6, 1e-8)
    m = tomps(href; cutoff = e^2)
    @printf("  h truncated to eps %.0e: rank %d, max err %.3e m\n", e, maxlinkdim(m), maximum(abs, fromtps(m) - href))
end

# check the MPO against M
let v = hmps_exact
    Av = fromtps(apply(A, v; alg = "naive", truncate = false))
    @printf("MPO check |A h - M h| / |M h| = %.2e\n", norm(vec(Av) - M * hsp) / norm(M * hsp))
end

# --- linsolve ------------------------------------------------------------------------------------
x0 = X0 == "warm" ? tomps(href; cutoff = 1e-4) : (b / norm(b)) * norm(hmps_exact)
x0 = x0 + 0.0 * x0
@printf("initial guess %s: rank %d\n", X0, maxlinkdim(x0))
x = x0
t_tot = 0.0
for sweep in 1:NSWEEPS
    global x, t_tot
    t = @elapsed x = linsolve(A, b, x; nsweeps = 1, cutoff = 1e-24, maxdim = MAXDIM, outputlevel = 0,
                               updater = direct_updater, updater_kwargs = (; maxdirect = 8000, krylov_kwargs = (; tol = 1e-12, maxiter = 40, krylovdim = 40, ishermitian = true)))
    t_tot += t
    X = fromtps(x)
    res = norm(M * vec(X) - rhs) / norm(rhs)
    err = norm(X - href) / norm(href)
    emax = maximum(abs, X - href)
    @printf("sweep %2d: %7.1f s (total %7.1f s), rank %3d, rel residual %.3e, rel error %.3e, max error %.3e m\n",
            sweep, t, t_tot, maxlinkdim(x), res, err, emax)
    flush(stdout)
    res < 1e-12 && break
end
