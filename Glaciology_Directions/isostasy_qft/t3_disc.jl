# Task 3a: disc load (FastIsostasy test1 geometry: W = 3000 km, R0 = 1000 km, H0 = 1 km, homogeneous
# ELVA) -- MPS/QFT pipeline vs dense FFTW vs analytic (Hankel) solution, accuracy, ranks and timings.
# usage: julia t3_disc.jl load=smooth|sharp w=20e3 nfft=7,8,9,10,11,12 nmps=7,...,12 eps=1e-6
#                         ordering=interleaved|sequential alg=naive|densitymatrix|zipup tag=...
include("isostasy.jl")
using Statistics
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1"))); FFTW.set_num_threads(1)

opts = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
loadname = get(opts, "load", "smooth")
w = parse(Float64, get(opts, "w", "20e3"))
parseints(s) = isempty(s) ? Int[] : parse.(Int, split(s, ","))
nfft = parseints(get(opts, "nfft", "7,8,9,10,11,12"))
nmps = parseints(get(opts, "nmps", "7,8,9,10,11,12"))
epsv = parse(Float64, get(opts, "eps", "1e-6"))
ord = Symbol(get(opts, "ordering", "interleaved"))
alg = get(opts, "alg", "naive"); malg = get(opts, "malg", "naive")
loadtci = parse(Int, get(opts, "loadtci", "13"))     # n >= loadtci: load MPS by TCI instead of dense SVD
tag = get(opts, "tag", "")
nc = 8                                                  # common sample lattice 256 x 256
p = Earth()
W = 3000e3; L = 2W; R0 = 1000e3; H0 = 1000.0
times = [100.0, 500.0, 1500.0, 5000.0, 10000.0, 50000.0, Inf]
loadf = loadname == "sharp" ? ((x, y) -> disc_sharp(x, y; R0 = R0, H0 = H0)) : ((x, y) -> disc_smooth(x, y; R0 = R0, H0 = H0, w = w))
hhat = loadname == "sharp" ? (k -> hhat_disc(k; R0 = R0, H0 = H0)) : (k -> hhat_smooth(k; R0 = R0, H0 = H0, w = w))
cut = epsv^2
@printf("disc test: load %s (w = %.0f km), eps = %.0e (cutoff %.0e), ordering %s, alg %s\n", loadname, w / 1e3, epsv, cut, ord, alg)
flush(stdout)

# analytic solution on the lattice row y = 0, x >= 0
xc = gridpoints(2^nc, W)
irow = 2^(nc - 1) .+ (0:2^(nc-1)-1)          # x = 0 .. W - dx_c
rs = xc[irow]
(Uan, ta) = timed(() -> analytic_profiles(rs, times, p; hhat = hhat))
@printf("analytic profiles: %.1f s; u(0) = %s m\n", ta, join([@sprintf("%.2f", Uan[1, i]) for i in 1:length(times)], ", "))
flush(stdout)
if loadname == "sharp"   # check the fixed rule against adaptive quadrature at a few radii
    for (r, it) in ((0.0, 3), (900e3, 5), (1.5e6, 7))
        @printf("  quadrature check r=%.0f km t=%s: fixed %.6f  adaptive %.6f\n", r / 1e3, times[it], Uan[argmin(abs.(rs .- r)), it], analytic_u(rs[argmin(abs.(rs .- r))], times[it], p))
        flush(stdout)
    end
end
rmask = rs .<= 1.8e6

results = Dict{String,Any}()
fftsamples = Dict{Int,Array{Float64,3}}()

println("\n== FFT (FFTW, 1 thread) ==")
flush(stdout)
@printf("%3s %8s %9s %9s %10s   %s\n", "n", "N", "t_fwd", "t_step", "mem(GB)", "max|u - analytic| (m) on r <= 1800 km, per output time")
flush(stdout)
for n in nfft
    N = 2^n
    x = gridpoints(N, W)
    sig = Float64[p.rho_ice * p.g * loadf(a, b) for a in x, b in x]
    P = plan_fft(complex.(sig); flags = FFTW.ESTIMATE)
    IP = plan_ifft(complex.(sig); flags = FFTW.ESTIMATE)
    (sh, tf) = timed(() -> P * complex.(sig))
    K = kgrid2(N, L)
    S = zeros(2^nc, 2^nc, length(times))
    tsteps = Float64[]
    errs = Float64[]
    for (it, t) in enumerate(times)
        (u, ts) = timed() do
            v = real.(IP * (sh .* mult_resp.(K, t, Ref(p))))
            corner_bc!(v)
        end
        push!(tsteps, ts)
        st = N ÷ 2^nc
        S[:, :, it] .= u[1:st:end, 1:st:end]
        push!(errs, maximum(abs.(S[irow, 2^(nc-1)+1, it] .- Uan[:, it])[rmask]))
        if n in nmps && n <= 12
            results["fft_u_$(n)_$(it)"] = u
        end
    end
    fftsamples[n] = S
    @printf("%3d %8d %8.3fs %8.3fs %10.2f   %s\n", n, N, tf, median(tsteps), 16 * N^2 * 3 / 1e9, join([@sprintf("%.3f", e) for e in errs], " "))
    flush(stdout)
    sig = nothing; sh = nothing; K = nothing; GC.gc()
end

println("\n== MPS (QFT MPO, ordering $ord, apply alg $alg, cutoff $cut) ==")
flush(stdout)
for n in nmps
    N = 2^n
    g = QGrid(2, n; ordering = ord)
    # load
    if n < loadtci
        x = gridpoints(N, W)
        sig = Float64[p.rho_ice * p.g * loadf(a, b) for a in x, b in x]
        (sm, tl) = timed(() -> array_to_mps(complex.(sig), g; cutoff = cut * 1e-2))
        sig = nothing
        lm = "svd"
    else
        f(idx) = p.rho_ice * p.g * loadf(-W + idx[1] * L / N, -W + idx[2] * L / N)
        (res, tl) = timed(() -> tci_mps(f, g; T = ComplexF64, tolerance = epsv * 1e-2, maxbonddim = 2000))
        sm = res[1]; truncate!(sm; cutoff = cut * 1e-2)
        lm = "tci"
    end
    (P, tq) = timed(() -> Pipeline(g; qcut = cut * 1e-4))
    (sh, tqa) = timed(() -> apply(P.Q, sm; alg = alg, cutoff = cut))
    @printf("n=%2d: load chi %4d (%s %.1fs) | QFT MPO chi %3d (build %.2fs) | sigma^ chi %4d (QFT apply %.2fs)\n",
            n, maxlinkdim(sm), lm, tl, maxlinkdim(P.Q), tq, maxlinkdim(sh), tqa)
    flush(stdout)
    S = zeros(2^nc, 2^nc, length(times))
    for (it, t) in enumerate(times)
        (m, tm) = timed(() -> mult_mps(k -> mult_resp(k, t, p), g, L; scale = 1 / N^2, tolerance = epsv * 1e-3, cutoff = cut * 1e-4))
        t0 = time()
        uh = apply(diag_mpo(m), sh; alg = malg, cutoff = cut)
        t1 = time()
        u = apply(P.Qa, uh; alg = alg, cutoff = cut)
        t2 = time()
        u = realpart(u; cutoff = cut)
        u = corner_bc(u, g; cutoff = cut)
        t3 = time()
        S[:, :, it] .= real.(coarse_samples(u, g, nc))
        ean = maximum(abs.(S[irow, 2^(nc-1)+1, it] .- Uan[:, it])[rmask])
        efft = NaN; efftmax = NaN
        if haskey(results, "fft_u_$(n)_$(it)")
            U = results["fft_u_$(n)_$(it)"]
            Um = real.(mps_to_array(u, g))
            efft = norm(Um - U) / norm(U); efftmax = maximum(abs.(Um - U))
        elseif haskey(fftsamples, maximum(nfft))
            Sf = fftsamples[maximum(nfft)][:, :, it]
            efft = norm(S[:, :, it] - Sf) / norm(Sf); efftmax = maximum(abs.(S[:, :, it] - Sf))
        end
        @printf("   t=%7s: mult chi %3d (TCI %.2fs) | uhat chi %4d u chi %4d | step %.2fs (mult %.2f, iQFT %.2f, re+bc %.2f) | vs FFT rel %.1e max %.1e m | vs analytic max %.3f m\n",
                isinf(t) ? "Inf" : string(Int(t)), maxlinkdim(m), tm, maxlinkdim(uh), maxlinkdim(u), t3 - t0, t1 - t0, t2 - t1, t3 - t2, efft, efftmax, ean)
        flush(stdout)
    end
    for k in collect(keys(results)); startswith(k, "fft_u_$(n)_") && delete!(results, k); end
    GC.gc()
end
println("done")
flush(stdout)
