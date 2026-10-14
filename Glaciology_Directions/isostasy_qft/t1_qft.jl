# Task 1: QFT MPO in 1D and 2D quantics -- correctness against FFTW, bond dimension vs bits and cutoff.
# usage: julia t1_qft.jl [nmax1=30] [nmax2=12] [quick=false]
include("qft.jl")
using FFTW
BLAS.set_num_threads(1); FFTW.set_num_threads(1)

opts = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
nmax1 = parse(Int, get(opts, "nmax1", "30"))
nmax2 = parse(Int, get(opts, "nmax2", "12"))
outdir = joinpath(@__DIR__, "results"); mkpath(outdir)
Random.seed!(1)

relerr(a, b) = norm(a - b) / norm(b)

println("== 0. conversion checks (point evaluation vs array) ==")
for (d, n, ord) in ((1, 6, :interleaved), (2, 4, :interleaved), (2, 4, :sequential))
    g = QGrid(d, n; ordering = ord)
    A = randn(ComplexF64, ntuple(_ -> npts(g), d))
    for rev in (false, true)
        psi = array_to_mps(A, g; rev = rev, cutoff = 0.0)
        E = MPSEval(psi)
        err = maximum(abs(evaluate(E, g, Tuple(I) .- 1; rev = rev) - A[I]) for I in CartesianIndices(A))
        err2 = maximum(abs.(mps_to_array(psi, g; rev = rev) - A))
        @printf("  d=%d n=%d %-12s rev=%-5s  point-eval err %.1e   roundtrip err %.1e\n", d, n, ord, rev, err, err2)
    end
end

println("\n== 1. 1D QFT MPO vs dense DFT matrix (n <= 10) ==")
for n in 2:10
    g = QGrid(1, n)
    Q = qft_mpo(g; cutoff = 1e-30)
    N = npts(g)
    # dense operator: rows = output (rev), cols = input (normal)
    T = Array(prod(Q), [s' for s in g.sites]..., g.sites...)
    M = zeros(ComplexF64, N, N)
    for J in CartesianIndices(T)
        sout = [J[i] for i in 1:n]; sin_ = [J[n+i] for i in 1:n]
        m = bits_to_idx(g, sout; rev = true)[1]; j = bits_to_idx(g, sin_)[1]
        M[m+1, j+1] = T[J]
    end
    F = [cis(-2pi * j * m / N) for m in 0:N-1, j in 0:N-1]
    @printf("  n=%2d  max|Q - DFT| = %.2e   maxlinkdim = %d\n", n, maximum(abs.(M - F)), maxlinkdim(Q))
end

println("\n== 2. 1D QFT MPO bond dimension vs n and cutoff ==")
# ITensors' cutoff is the discarded squared weight: cutoff = eps^2 <-> relative error ~ eps
epss = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
cutoffs = epss .^ 2
rows1 = []
@printf("  %4s", "n"); for c in epss; @printf("  eps=%.0e", c); end; @printf("   t_build(eps=1e-14)\n")
for n in [4:2:20; 24; 30][[4:2:20; 24; 30] .<= nmax1]
    g = QGrid(1, n)
    @printf("  %4d", n)
    chis = Int[]; tb = 0.0
    for c in cutoffs
        Q, tb = timed(() -> qft_mpo(g; cutoff = c))
        push!(chis, maxlinkdim(Q)); @printf("  %9d", maxlinkdim(Q))
    end
    @printf("   %.2f s\n", tb)
    push!(rows1, (n, chis))
end

println("\n== 3. 1D accuracy on vectors vs FFTW ==")
println("  random vectors (exact MPS, bond 2^(n/2))")
for n in (8, 10, 12, 14), c in (1e-8, 1e-16, 1e-24, 1e-28)
    g = QGrid(1, n); N = npts(g)
    x = randn(ComplexF64, N)
    psi = array_to_mps(x, g; cutoff = 0.0)
    Q = qft_mpo(g; cutoff = c)
    (y, t) = timed(() -> apply(Q, psi; alg = "naive", cutoff = 1e-30))
    err = relerr(vec(mps_to_array(y, g; rev = true)), fft(x))
    @printf("  n=%2d cutoff=%.0e  chi_Q=%2d  rel err %.2e  (apply %.2f s, out chi %d)\n", n, c, maxlinkdim(Q), err, t, maxlinkdim(y))
end
println("  smooth/structured vectors (Gaussian bump + sine + step), MPS from dense SVD cutoff 1e-15")
for n in (12, 16, 20, 24)
    g = QGrid(1, n); N = npts(g)
    xs = ((0:N-1) .+ 0.5) ./ N
    x = ComplexF64.(exp.(-((xs .- 0.3) ./ 0.05) .^ 2) .+ 0.3 .* sin.(2pi * 7 .* xs) .+ (xs .> 0.7))
    psi = array_to_mps(x, g; cutoff = 1e-30)
    (xf, tf) = timed(() -> fft(x))
    @printf("  n=%2d FFTW %.3f s\n", n, tf)
    for c in (1e-12, 1e-20, 1e-28)
        Q = qft_mpo(g; cutoff = c)
        (y, t) = timed(() -> apply(Q, psi; alg = "naive", cutoff = c))
        err = relerr(vec(mps_to_array(y, g; rev = true)), xf)
        @printf("  n=%2d chi_in=%3d cutoff=%.0e chi_Q=%2d  rel err %.2e  chi_out %3d  apply %.2f s\n",
                n, maxlinkdim(psi), c, maxlinkdim(Q), err, maxlinkdim(y), t)
    end
    # gate-by-gate circuit on the MPS, no MPO
    (y2, t2) = timed(() -> qft_circuit(psi, g; cutoff = 1e-28))
    @printf("        circuit (gate by gate, cutoff 1e-28): rel err %.2e chi_out %3d  %.2f s\n",
            relerr(vec(mps_to_array(y2, g; rev = true)), xf), maxlinkdim(y2), t2)
    # inverse
    Q = qft_mpo(g; cutoff = 1e-28)
    yb = apply(adjoint_mpo(Q), apply(Q, psi; alg = "naive", cutoff = 1e-28); alg = "naive", cutoff = 1e-28) / N
    @printf("        inverse: |Q^+ Q x / N - x| / |x| = %.2e\n", relerr(vec(mps_to_array(yb, g)), x))
end

println("\n== 4. 2D QFT MPO bond dimension (interleaved vs sequential) ==")
@printf("  %4s  %14s %14s %14s %14s %14s %14s\n", "n", "int(eps=1e-4)", "int(1e-8)", "int(1e-12)", "seq(1e-4)", "seq(1e-8)", "seq(1e-12)")
for n in 3:nmax2
    chis = Int[]; ts = Float64[]
    for ord in (:interleaved, :sequential), c in (1e-8, 1e-16, 1e-24)
        g = QGrid(2, n; ordering = ord)
        Q, t = timed(() -> qft_mpo(g; cutoff = c))
        push!(chis, maxlinkdim(Q)); push!(ts, t)
    end
    @printf("  %4d ", n); for i in 1:6; @printf(" %8d(%4.1fs)", chis[i], ts[i]); end; println()
end

println("\n== 5. 2D accuracy vs FFTW ==")
for ord in (:interleaved, :sequential)
    g = QGrid(2, 5; ordering = ord); N = npts(g)
    A = randn(ComplexF64, N, N)
    psi = array_to_mps(A, g; cutoff = 0.0)
    Q = qft_mpo(g; cutoff = 1e-28)
    y = apply(Q, psi; alg = "naive", cutoff = 1e-30)
    @printf("  random 32x32 %-12s chi_Q=%3d rel err %.2e\n", ord, maxlinkdim(Q), relerr(mps_to_array(y, g; rev = true), fft(A)))
end
for n in (8, 10, 12), ord in (:interleaved, :sequential)
    g = QGrid(2, n; ordering = ord); N = npts(g)
    xs = ((0:N-1) .+ 0.5) ./ N
    A = ComplexF64[exp(-((x - 0.4)^2 + (y - 0.55)^2) / 0.02) * (1 + 0.2 * sin(2pi * 3 * x)) for x in xs, y in xs]
    psi = array_to_mps(A, g; cutoff = 1e-28)
    (Af, tf) = timed(() -> fft(A))
    for c in (1e-12, 1e-20, 1e-28)
        Q = qft_mpo(g; cutoff = c)
        (y, t) = timed(() -> apply(Q, psi; alg = "naive", cutoff = c))
        @printf("  smooth %4dx%-4d %-12s chi_in=%3d cutoff=%.0e chi_Q=%3d rel err %.2e chi_out %3d apply %.2f s\n",
                N, N, ord, maxlinkdim(psi), c, maxlinkdim(Q), relerr(mps_to_array(y, g; rev = true), Af), maxlinkdim(y), t)
    end
end
println("done")
