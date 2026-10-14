# Cost/accuracy of the Newton linear solve J dH = -R (1D Halfar, dt = 10 a) with the direct local
# solver, vs truncation of the iterate (lin_cutoff, lin_maxdim) and number of sweeps, against a
# sparse direct solve with the same right-hand side.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_linsolve2.jl bits=12,14,16
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "12,14,16", "dt" => "2.5")
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
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    h1 = mps_to_array(H1, g); He = Avg * h1; gg = Df * h1
    Js = I - dt * Db * (spdiagm(0 => 3G .* He .^ 5 .* gg .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* gg .^ 3) * Avg)
    xs = Js \ (-mps_to_array(R, g))
    println(@sprintf("\n2^%d points, dt %.1f: chi(J) %d, chi(R) %d", nbits, dt, maxlinkdim(J), maxlinkdim(R))); flush(stdout)
    for lc in (1e-14, 1e-12, 1e-10), lm in (16, 32), ns in (1, 2)
        st = SolverSettings(lin_cutoff = lc, lin_maxdim = lm, lin_retries = 0)
        _linsolve_once(J, -1.0 * R, -1.0 * R, st, false, 1)   # warm-up / compile
        t = @elapsed x = _linsolve_once(J, -1.0 * R, -1.0 * R, st, false, ns)
        println(@sprintf("   lin_cutoff %.0e maxdim %2d nsweeps %d: %6.2f s  chi(dH) %2d  res %.1e  err vs sparse %.1e",
                         lc, lm, ns, t, maxlinkdim(x), residual_norm(J, x, -1.0 * R), norm(mps_to_array(x, g) - xs) / norm(xs)))
        flush(stdout)
    end
end
