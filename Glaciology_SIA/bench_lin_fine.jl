# Linear solve of the complementarity Newton system J dH = -Phi near the EISMINT steady state on a
# fine 1D grid (state from a run's output, exactly prolonged to nbits), for several linsolve settings
# and iterative refinement. Reports the exact relative residual and the time.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_lin_fine.jl file=output/<run>.h5 nbits=18 dt=100
using Printf, HDF5, SparseArrays, LinearAlgebra
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

opts = Dict("file" => "", "nbits" => "18", "dt" => "100", "maxdims" => "128,256", "sweeps" => "4,16", "cutoffs" => "1e-16,1e-20",
            "refine" => "4", "tols" => "", "maxsweeps" => "64", "diag" => "0")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
dt = parse(Float64, opts["dt"])
Hm, nbc, ord = h5open(opts["file"], "r") do f
    read(f, "H_mps", MPS), read(attributes(f)["nbits"]), Symbol(read(attributes(f)["ordering"]))
end
p = SIAParams(); eb = EISMINTMovingMargin(p = p); L = eb.L
gc = QuanticsGrid(nbc, ord, (-L, L))
nb = parse(Int, opts["nbits"])
g = QuanticsGrid(nb, ord, (-L, L))
H = nb > nbc ? prolong_mps(Hm, gc, g) : replace_siteinds(Hm, g.sites)
ops = SIAOperators(g)
st0 = SolverSettings(cutoff = 1e-20, op_cutoff = 1e-22, prod_cutoff = 1e-24, prod_maxdim = 256, complementarity = true)
kw = (; cutoff = st0.cutoff, maxdim = st0.maxdim)
a_mb = function_to_mps(x -> eismint_mb(eb, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
rhs0 = lincomb([1.0, dt], [H, a_mb]; kw...)
F = face_fields(H, nothing, ops, p, st0)
R = lincomb([1.0, -dt, -1.0], [H, flux_divergence(F, ops, st0), rhs0]; kw...)
mask = indicator_mps((h, r) -> h - r, (H, R), g; tolerance = st0.ncp_tol, maxbonddim = st0.maxdim)
Phi = lincomb([1.0, 1.0, -1.0], [hadamard(mask, R; prod_kw(st0)...), H, hadamard(mask, H; prod_kw(st0)...)]; kw...)
Jac = mpo_product(diag_mpo(mask), jacobian_mpo(F, ops, p, st0); cutoff = st0.op_cutoff, maxdim = st0.op_maxdim)
J = mpo_lincomb([1.0, -dt], [ops.I, Jac]; cutoff = st0.op_cutoff, maxdim = st0.op_maxdim)
b = -1.0 * Phi
@printf("2^%d, dt = %g: chi(H) %d chi(J) %d chi(Phi) %d |Phi|/|rhs| %.2e\n", nb, dt, maxlinkdim(H), maxlinkdim(J), maxlinkdim(Phi), norm(Phi) / norm(rhs0))
# exact reference: J is tridiagonal in 1D (3-point Mahaffy stencil); its diagonals from J applied to
# the three period-3 comb vectors, then a sparse direct solve
N = 2^nb
Jv = [mps_to_array(mps_apply(J, array_to_mps(Float64.(mod.(0:N-1, 3) .== k), g; cutoff = 1e-30); cutoff = 1e-30, maxdim = typemax(Int)), g) for k in 0:2]
Ii = Int[]; Jj = Int[]; Vv = Float64[]
for j in 1:N, i in max(1, j - 1):min(N, j + 1)
    push!(Ii, i); push!(Jj, j); push!(Vv, Jv[mod(j - 1, 3) + 1][i])
end
Js = sparse(Ii, Jj, Vv, N, N)
bv = mps_to_array(b, g)
chk = norm(Js * bv - mps_to_array(mps_apply(J, b; cutoff = 1e-30, maxdim = typemax(Int)), g)) / norm(Js * bv)
xe = Js \ bv
@printf("sparse J check %.1e, cond_1 estimate %.1e, |x_exact| %.3e\n", chk, cond(Js, 1), norm(xe))
ferr(x) = norm(mps_to_array(x, g) - xe) / norm(xe)
if opts["diag"] == "1"
    # where is the exact Newton step large, and what do H, R, mask and the J rows look like there
    Hv = mps_to_array(H, g); Rv = mps_to_array(R, g); mv = mps_to_array(mask, g); xs = [-L + (i - 0.5) * 2L / N for i in 1:N]
    idx = sortperm(abs.(xe), rev = true)[1:12]
    println("  i        x [km]     H [m]        R [m]        mask  x_exact      J[i,i-1]     J[i,i]       J[i,i+1]")
    for i in sort(idx)
        @printf("%7d  %9.3f  %11.4e  %11.4e  %4.2f  %11.4e  %11.4e  %11.4e  %11.4e\n", i, xs[i] / 1e3, Hv[i], Rv[i], mv[i], xe[i],
                i > 1 ? Js[i, i-1] : 0.0, Js[i, i], i < N ? Js[i, i+1] : 0.0)
    end
    println("fraction of |x|^2 within 2 km of the margins: ", sum(abs2, xe[abs.(abs.(xs) .- 656e3) .< 2e3]) / sum(abs2, xe))
    neg = findall(<(0), diag(Js)); println("negative diagonal entries: ", length(neg), neg[1:min(end, 5)])
    exit()
end
println("maxdim  sweeps  cutoff   refine   res        tries  chi(x)   t [s]   fwd err"); flush(stdout)
refine = parse(Int, opts["refine"])
for md in (refine < 0 ? Int[] : parse.(Int, split(opts["maxdims"], ","))), ns in parse.(Int, split(opts["sweeps"], ",")), co in parse.(Float64, split(opts["cutoffs"], ","))
    for rf in (0, refine)
        st = SolverSettings(cutoff = 1e-20, op_cutoff = 1e-22, prod_cutoff = 1e-24, prod_maxdim = 256, complementarity = true, lin_maxdim = md, nsweeps = ns, lin_cutoff = co, lin_res_tol = 1e-12, lin_retries = 0, lin_refine = rf, lin_refine_tol = 1e-9)
        t = @elapsed x, (res, tries) = _linsolve(J, b, b, st, false)
        @printf("%6d  %6d  %6.0e  %6d   %.2e  %5d  %6d  %6.1f  %.2e\n", md, ns, co, rf, res, tries, maxlinkdim(x), t, ferr(x))
        flush(stdout)
    end
end

# tolerance-controlled sweeping (stop on the exact residual), at most 64 sweeps
for md in parse.(Int, split(opts["maxdims"], ",")), co in parse.(Float64, split(opts["cutoffs"], ",")), tl in (isempty(opts["tols"]) ? Float64[] : parse.(Float64, split(opts["tols"], ",")))
    st = SolverSettings(cutoff = 1e-20, op_cutoff = 1e-22, prod_cutoff = 1e-24, prod_maxdim = 256, complementarity = true, lin_maxdim = md, lin_cutoff = co, lin_tol = tl, lin_maxsweeps = parse(Int, opts["maxsweeps"]))
    t = @elapsed x, (res, nsw) = _linsolve(J, b, b, st, false)
    @printf("tol %.0e  maxdim %d  cutoff %.0e: res %.2e after %d sweeps, chi(x) %d, %.1f s, fwd err %.2e\n", tl, md, co, res, nsw, maxlinkdim(x), t, ferr(x))
    flush(stdout)
end
