# Newton system J dH = -R (1D Halfar, dt = 2.5 a) at high resolution: the DMRG-like linsolve
# stalls for very ill-conditioned J. Compare (A) more sweeps and (B) iterative refinement
# x <- x + linsolve(J, r), r = b - J x formed exactly, against a sparse direct solve.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_refine.jl bits=14,16,18
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "14,16,18", "dt" => "2.5")
for a in ARGS
    k, v = split(a, "="); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
dt = parse(Float64, opts["dt"])
p = SIAParams(); hb = Halfar(d = 1, p = p); G = Gamma(p); t0 = halfar_t0(hb)

for nbits in bits
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3)); ops = SIAOperators(g)
    st0 = SolverSettings()
    H0 = function_to_mps(x -> halfar_H(hb, t0, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    H1 = function_to_mps(x -> halfar_H(hb, t0 + dt, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    F1 = face_fields(H1, nothing, ops, p, st0)
    R = lincomb([1.0, -dt, -1.0], [H1, flux_divergence(F1, ops, st0), H0]; cutoff = 1e-24)
    J = mpo_lincomb([1.0, -dt], [ops.I, jacobian_mpo(F1, ops, p, st0)]; cutoff = st0.op_cutoff, maxdim = 512)
    b = -1.0 * R
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    h1 = mps_to_array(H1, g); He = Avg * h1; gg = Df * h1
    Js = I - dt * Db * (spdiagm(0 => 3G .* He .^ 5 .* gg .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* gg .^ 3) * Avg)
    bs = mps_to_array(b, g)
    xs = Js \ bs
    println(@sprintf("\n2^%d points, dt %.1f: chi(J) %d, chi(b) %d", nbits, dt, maxlinkdim(J), maxlinkdim(b))); flush(stdout)
    err(x) = norm(mps_to_array(x, g) - xs) / norm(xs)
    st = SolverSettings(lin_cutoff = 1e-12, lin_maxdim = 64, lin_retries = 0)
    _linsolve_once(J, b, b, st, false, 1)   # compile
    for ns in (2, 8, 32)
        t = @elapsed x = _linsolve_once(J, b, b, st, false, ns)
        println(@sprintf("   (A) nsweeps %2d: %6.2f s  chi %2d  res %.1e  err %.1e", ns, t, maxlinkdim(x), residual_norm(J, x, b), err(x)))
        flush(stdout)
    end
    t = @elapsed x = _linsolve_once(J, b, b, st, false, 2)
    for k in 1:8
        t += @elapsed begin
            r = lincomb([1.0, -1.0], [b, mps_apply(J, x; cutoff = 1e-24, maxdim = 512)]; cutoff = 1e-20, maxdim = 256)
            dx = _linsolve_once(J, r, r, st, false, 2)
            x = lincomb([1.0, 1.0], [x, dx]; cutoff = 1e-14, maxdim = 128)
        end
        println(@sprintf("   (B) refinement %d: %6.2f s  chi(x) %2d chi(r) %3d  res %.1e  err %.1e", k, t, maxlinkdim(x), maxlinkdim(r), residual_norm(J, x, b), err(x)))
        flush(stdout)
    end
end
