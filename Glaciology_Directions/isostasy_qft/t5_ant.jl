# Task 3b: Antarctic grounded-ice load (BedMachine 8 km, padded to 1024^2 -> 8192 km periodic box).
# MPS/QFT pipeline vs dense FFTW, and piecewise-constant refinement beyond the data resolution
# (interleaved ordering: appended finest bits). usage: julia t5_ant.jl eps=1e-3 ordering=interleaved nup=0,2,4
include("isostasy.jl")
using Statistics
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1"))); FFTW.set_num_threads(1)
opts = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
epsv = parse(Float64, get(opts, "eps", "1e-3"))
ord = Symbol(get(opts, "ordering", "interleaved"))
nups = parse.(Int, split(get(opts, "nup", "0"), ","))
cut = epsv^2
p = Earth()
H, W = antarctic_load()
L = 2W
@printf("ANT: eps %.0e ordering %s, W = %.0f km, max H %.0f m\n", epsv, ord, W / 1e3, maximum(H)); flush(stdout)
n0 = 10; N0 = 2^n0
sig = p.rho_ice * p.g .* H
K0 = kgrid2(N0, L)
(sh0, tf) = timed(() -> fft(complex.(sig)))
times = [1000.0, 10000.0, Inf]
ufft = Dict{Float64,Matrix{Float64}}()
for t in times
    (u, ts) = timed(() -> real.(ifft(sh0 .* mult_resp.(K0, t, Ref(p)))))
    ufft[t] = u
    @printf("FFTW n=10 t=%s: %.2f s (fwd %.2f s); u range %.1f .. %.1f m\n", t, ts, tf, extrema(u)...); flush(stdout)
end
g = QGrid(2, n0; ordering = ord)
(sm, tl) = timed(() -> array_to_mps(complex.(sig), g; cutoff = cut * 1e-2))
truncate!(sm; cutoff = cut * 1e-2)
@printf("load MPS n=10: chi %d, params %d (dense %d), svd %.1fs\n", maxlinkdim(sm), nparams(sm), N0^2, tl); flush(stdout)
(P, tq) = timed(() -> Pipeline(g; qcut = cut * 1e-4))
@printf("QFT MPO chi %d (%.1fs)\n", maxlinkdim(P.Q), tq); flush(stdout)
(sh, tqa) = timed(() -> apply(P.Q, sm; alg = "naive", cutoff = cut))
@printf("sigma^ chi %d, QFT apply %.1fs, params %d\n", maxlinkdim(sh), tqa, nparams(sh)); flush(stdout)
for t in times
    (m, tm) = timed(() -> mult_mps(k -> mult_resp(k, t, p), g, L; scale = 1 / N0^2, tolerance = epsv * 1e-3, cutoff = cut * 1e-4))
    t0 = time()
    uh = apply(diag_mpo(m), sh; alg = "naive", cutoff = cut)
    u = apply(P.Qa, uh; alg = "naive", cutoff = cut)
    u = corner_bc(realpart(u; cutoff = cut), g; cutoff = cut)
    ts = time() - t0
    U = ufft[t]; U = U .- (U[1, 1] + U[1, end] + U[end, 1] + U[end, end]) / 4
    Um = real.(mps_to_array(u, g))
    @printf("t=%s: mult chi %d | uhat chi %d u chi %d (params %d) | MPS %.1fs | vs FFT: rel L2 %.2e, max %.3f m (max|u| %.1f m)\n",
            t, maxlinkdim(m), maxlinkdim(uh), maxlinkdim(u), nparams(u), ts, norm(Um - U) / norm(U), maximum(abs.(Um - U)), maximum(abs.(U))); flush(stdout)
end
# ---- refinement beyond the data (interleaved only) ----
if ord == :interleaved
    t = 10000.0
    E12 = nothing
    for nup in nups
        nup == 0 && continue
        n = n0 + nup; N = 2^n
        gf = QGrid(2, n; ordering = ord)
        sf = upsample_mps(sm, g, gf)
        (Pf, tq) = timed(() -> Pipeline(gf; qcut = cut * 1e-4))
        t0 = time()
        shf = apply(Pf.Q, sf; alg = "naive", cutoff = cut)
        (m, tm) = timed(() -> mult_mps(k -> mult_resp(k, t, p), gf, L; scale = 1 / N^2, tolerance = epsv * 1e-3, cutoff = cut * 1e-4))
        uh = apply(diag_mpo(m), shf; alg = "naive", cutoff = cut)
        u = apply(Pf.Qa, uh; alg = "naive", cutoff = cut)
        u = corner_bc(realpart(u; cutoff = cut), gf; cutoff = cut)
        ts = time() - t0
        S = real.(coarse_samples(u, gf, n0))          # samples at the coarse vertices (i 2^nup, j 2^nup)
        # FFT reference at n=10 with the same (piecewise-constant) load sampled at the same points is the n=10 solution only
        # up to the sub-cell offset, so compare with the n=n0 FFT solution (smooth, differences ~ cell-scale effects)
        U = ufft[t]; U = U .- (U[1, 1] + U[1, end] + U[end, 1] + U[end, end]) / 4
        @printf("refined n=%d (N=%d, %.1e points, dense would need %.1f GB per field): chi: Q %d, load %d, sigma^ %d, uhat %d, u %d | total %.1fs | u vs FFT(n=10) on common lattice rel %.2e | params(u) %d\n",
                n, N, Float64(N)^2, 8 * Float64(N)^2 / 1e9, maxlinkdim(Pf.Q), maxlinkdim(sf), maxlinkdim(shf), maxlinkdim(uh), maxlinkdim(u), ts,
                norm(S - U) / norm(U), nparams(u)); flush(stdout)
    end
end
println("done")
