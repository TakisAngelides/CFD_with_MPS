# Consistency checks of the quantics machinery against dense arrays (small grids)
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch tests.jl

using LinearAlgebra, SparseArrays, Random, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")
Random.seed!(1)

check(name, err, tol) = println(@sprintf("%-55s %.2e  %s", name, err, err < tol ? "ok" : "FAIL (tol $(tol))"))

# 1) shifts / round trips for d = 1, 2, 3 and both orderings
for d in 1:3, ord in (:interleaved, :sequential)
    n = d == 3 ? 3 : 4
    g = QuanticsGrid(n, ord, ntuple(_ -> (0.0, 1.0), d)...)
    A = randn(ntuple(_ -> 2^n, d)...)
    m = array_to_mps(A, g)
    check("d=$d $ord array round trip", maximum(abs, mps_to_array(m, g) - A), 1e-10)
    err = 0.0
    for a in 1:d, dir in (1, -1)
        ref = zeros(size(A)...)
        for I in CartesianIndices(A)
            J = collect(Tuple(I)); J[a] += dir
            (1 <= J[a] <= 2^n) && (ref[I] = A[J...])
        end
        err = max(err, maximum(abs, mps_to_array(apply(shift_mpo(g, a, dir), m), g) - ref))
        per = circshift(A, ntuple(b -> b == a ? -dir : 0, d))
        err = max(err, maximum(abs, mps_to_array(apply(shift_mpo(g, a, dir; pbc = true), m), g) - per))
    end
    check("d=$d $ord shifts (open + periodic)", err, 1e-10)
    E = MPSEvaluator(m)
    check("d=$d $ord point evaluation", abs(E(idx_to_bits(g, ntuple(identity, d))) - A[(ntuple(identity, d) .+ 1)...]), 1e-10)
end

# 2) TCI and integrals in 2D
g = QuanticsGrid(9, :interleaved, (-1.0, 1.0), (-1.0, 1.0))
f(x, y) = exp(-(x^2 + 2y^2) / 0.1)
m = function_to_mps(f, g; tolerance = 1e-10)
check("2D TCI max error", maximum(abs, mps_to_array(m, g) - [f(x, y) for x in coords(g, 1), y in coords(g, 2)]), 1e-8)
check("2D integral (relative, incl. quadrature error)", abs(mps_sum(g, m) * prod(g.h) - pi * 0.1 / sqrt(2)) / (pi * 0.1 / sqrt(2)), 1e-4)
check("Hadamard power", maximum(abs, mps_to_array(hadamard_pow(m, 5; cutoff = 1e-15), g) - mps_to_array(m, g) .^ 5), 1e-6)

# 3) SIA operator and implicit linear solve against sparse dense reference (1D Halfar profile)
p = SIAParams()
hb = Halfar(d = 1, p = p)
g = QuanticsGrid(10, :interleaved, (-1200e3, 1200e3))
ops = SIAOperators(g)
st = SolverSettings()
H = function_to_mps(x -> halfar_H(hb, halfar_t0(hb), abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
Hd = mps_to_array(H, g); N = length(Hd); h = g.h[1]
D = face_diffusivities(H, nothing, ops, p, st)
Dd = mps_to_array(D[1], g)
He = [(Hd[i] + (i < N ? Hd[i+1] : 0.0)) / 2 for i in 1:N]
sx = [((i < N ? Hd[i+1] : 0.0) - Hd[i]) / h for i in 1:N]
Dref = Gamma(p) .* He .^ 5 .* sx .^ 2
check("SIA face diffusivity (relative)", maximum(abs, Dd - Dref) / maximum(Dref), 1e-2)
Df = spdiagm(0 => -ones(N), 1 => ones(N - 1)) / h
Db = spdiagm(0 => ones(N), -1 => -ones(N - 1)) / h
Lmat = Db * spdiagm(0 => Dd) * Df
L = diffusion_mpo(D, ops, st)
check("L[D] H vs sparse (relative)", maximum(abs, mps_to_array(apply(L, H), g) - Lmat * Hd) / maximum(abs, Lmat * Hd), 1e-4)
dt = 100.0
Amp = mpo_lincomb([1.0, -dt], [ops.I, L]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
x = linsolve(Amp, H, H; nsweeps = st.nsweeps, cutoff = st.cutoff, maxdim = st.maxdim, outputlevel = 0,
             updater_kwargs = (; ishermitian = true, tol = st.krylov_tol, maxiter = 30, krylovdim = 30))
xref = (I - dt * Lmat) \ Hd
check("linsolve (I - dt L) x = H vs sparse (relative)", maximum(abs, mps_to_array(x, g) - xref) / maximum(xref), 1e-3)
xd = mps_to_array(x, g)
rd = norm(Hd - (I - dt * Lmat) * xd) / norm(Hd)
# residual_norm expands <b,b> - 2<b|A|x> + <Ax|Ax>, so a residual of relative size 1e-4 is only
# known to a few per cent (cancellation); it is used as a retry trigger at the 2e-2 level
check("exact residual_norm vs dense (relative to dense value $(round(rd, sigdigits=3)))", abs(residual_norm(Amp, x, H) - rd) / rd, 1e-1)
println("chi(L) = ", maxlinkdim(L), ", chi(A) = ", maxlinkdim(Amp), ", chi(x) = ", maxlinkdim(x))

# 4) Newton Jacobian vs finite differences of a plain-array implementation of the same stencil
#    (MPS noise would swamp a finite difference of the MPS residual itself)
shiftA(A, a, dir) = begin                       # B[i] = A[i + dir along a], zero outside
    B = zeros(size(A)); n_ = size(A, a)
    for I in CartesianIndices(A)
        J = collect(Tuple(I)); J[a] += dir
        (1 <= J[a] <= n_) && (B[I] = A[J...])
    end
    B
end
function F_dense(Hd, h, p)
    d = ndims(Hd); out = zeros(size(Hd))
    for a in 1:d
        Ha = (Hd + shiftA(Hd, a, 1)) / 2
        gr = [c == a ? (shiftA(Hd, a, 1) - Hd) / h :
              (((shiftA(Hd, c, 1) - shiftA(Hd, c, -1)) / (2h)) + shiftA((shiftA(Hd, c, 1) - shiftA(Hd, c, -1)) / (2h), a, 1)) / 2
              for c in 1:d]
        flux = Gamma(p) .* Ha .^ (p.n_glen + 2) .* sum(gc .^ 2 for gc in gr) .^ ((p.n_glen - 1) ÷ 2) .* gr[a]
        out += (flux - shiftA(flux, a, -1)) / h
    end
    out
end
st_exact = SolverSettings(cutoff = 1e-15, maxdim = 1024, op_cutoff = 1e-25, op_maxdim = 4096, prod_cutoff = 1e-28, prod_maxdim = 1024)
for d in (1, 2)
    st = st_exact          # near-exact truncation, so the test checks the derivation not truncation
    hb2 = Halfar(d = d, p = p)
    g2 = QuanticsGrid(d == 1 ? 10 : 7, :interleaved, ntuple(_ -> (-1200e3, 1200e3), d)...)
    ops2 = SIAOperators(g2)
    H2 = function_to_mps((x...) -> halfar_H(hb2, halfar_t0(hb2), sqrt(sum(abs2, x))), g2; tolerance = 1e-8, maxbonddim = 128)
    v = function_to_mps((x...) -> 100 * exp(-sum(abs2, x) / (300e3)^2), g2; tolerance = 1e-10)
    H2d, vd = mps_to_array(H2, g2), mps_to_array(v, g2)
    F0 = face_fields(H2, nothing, ops2, p, st)
    check("d=$d flux divergence vs dense (relative)", maximum(abs, mps_to_array(flux_divergence(F0, ops2, st), g2) - F_dense(H2d, g2.h[1], p)) / maximum(abs, F_dense(H2d, g2.h[1], p)), 1e-4)
    J = jacobian_mpo(F0, ops2, p, st)
    Jv = mps_to_array(mps_apply(J, v; cutoff = 1e-28, maxdim = 4096), g2)
    eps_ = 1e-4
    fd = (F_dense(H2d + eps_ * vd, g2.h[1], p) - F_dense(H2d - eps_ * vd, g2.h[1], p)) / (2eps_)
    check("d=$d Jacobian J v vs dense central FD (relative)", maximum(abs, Jv - fd) / maximum(abs, fd), 1e-4)
    println("chi(J) = ", maxlinkdim(J))
end

# 5) exact prolongation (grid sequencing) vs the dense multilinear interpolant, d = 1, 2, both orderings
for d in (1, 2), ord in (:interleaved, :sequential)
    nc, nf = d == 1 ? (5, 8) : (3, 5)
    gc = QuanticsGrid(nc, ord, ntuple(_ -> (-1.0, 1.0), d)...)
    gf = QuanticsGrid(nf, ord, ntuple(_ -> (-1.0, 1.0), d)...)
    Ac = randn(ntuple(_ -> 2^nc, d)...)
    Hc = array_to_mps(Ac, gc; cutoff = 1e-16)
    ref = map(Iterators.product(ntuple(a -> coords(gf, a), d)...)) do x
        il = ntuple(a -> clamp(floor(Int, (x[a] - gc.lo[a]) / gc.h[a] - 0.5), 0, 2^nc - 2), d)
        w = ntuple(a -> clamp((x[a] - coord(gc, a, il[a])) / gc.h[a], 0.0, 1.0), d)
        sum(prod(c[a] == 1 ? w[a] : 1 - w[a] for a in 1:d) * Ac[(il .+ c .+ 1)...] for c in Iterators.product(ntuple(_ -> (0, 1), d)...))
    end
    Hf = prolong_mps(Hc, gc, gf; cutoff = 1e-24)
    check("d=$d $ord exact prolongation 2^$nc -> 2^$nf (chi $(maxlinkdim(Hf)))", maximum(abs, mps_to_array(Hf, gf) - ref) / maximum(abs, ref), 1e-12)
end
