# Diagnosis of the 2^16 Newton system: conditioning of the sparse J (largest/smallest singular
# value by power/inverse iteration on J'J) and convergence of the MPS solve with many sweeps.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_cond16.jl bits=14,16
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "14,16", "dt" => "2.5")
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
    Fs = lu(Js); Fst = lu(sparse(Js'))
    v = randn(N); for _ in 1:50; v = Js' * (Js * v); v /= norm(v); end
    smax = sqrt(norm(Js' * (Js * v)))
    w = randn(N); for _ in 1:50; w = Fs \ (Fst \ w); w /= norm(w); end
    smin = 1 / sqrt(norm(Fs \ (Fst \ w)))
    bs = mps_to_array(b, g); xs = Fs \ bs
    # where does the smallest singular vector live?
    imax = argmax(abs.(w))
    println(@sprintf("\n2^%d: sigma_max %.2e sigma_min %.2e cond %.2e; smallest singular vector peaks at x = %.1f km (H = %.1f m); |x_sparse|/|b| %.1e",
                     nbits, smax, smin, smax / smin, coord(g, 1, imax - 1) / 1e3, h1[imax], norm(xs) / norm(bs)))
    # MPS: residual of the sparse solution in the MPS operator, and many sweeps
    xsm = array_to_mps(xs, g; cutoff = 1e-20)
    println(@sprintf("   sparse solution as MPS: chi %d, MPS residual %.1e", maxlinkdim(xsm), residual_norm(J, xsm, b)))
    for (lc, lm, ns) in ((1e-12, 64, 4), (1e-12, 64, 16), (1e-14, 128, 16))
        st = SolverSettings(lin_cutoff = lc, lin_maxdim = lm, lin_retries = 0)
        t = @elapsed x = _linsolve_once(J, b, b, st, false, ns)
        t2 = @elapsed x2 = _linsolve_once(J, b, xsm, st, false, 2)
        println(@sprintf("   lin_cutoff %.0e maxdim %3d nsweeps %2d: %6.2f s chi %2d res %.1e err %.1e | from sparse solution: res %.1e err %.1e",
                         lc, lm, ns, t, maxlinkdim(x), residual_norm(J, x, b), norm(mps_to_array(x, g) - xs) / norm(xs),
                         residual_norm(J, x2, b), norm(mps_to_array(x2, g) - xs) / norm(xs)))
        flush(stdout)
    end
end
