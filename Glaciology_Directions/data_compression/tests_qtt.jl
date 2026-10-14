# tests_qtt.jl -- correctness checks of qttlib.jl (layout, TT-SVD round trip, point evaluation,
# agreement with Glaciology_SIA/quantics.jl conventions, wavelet perfect reconstruction, fill).
using Random, Test
include("qttlib.jl")
include(joinpath(@__DIR__, "..", "..", "Glaciology_SIA", "quantics.jl"))

Random.seed!(1)
BLAS.set_num_threads(Threads.nthreads())

@testset "layout + TT-SVD" begin
    for ord in (:interleaved, :sequential), n in (3, 6)
        N = 2^n
        A = randn(N, N)
        @test from_chain(to_chain(A, ord), n, ord) == A
        q = ttsvd(to_chain(A, ord); eps = 1e-13)
        B = qtt_to_array(q, n, ord)
        @test maximum(abs.(A - B)) < 1e-10
        for _ in 1:20
            i, j = rand(0:N-1), rand(0:N-1)
            @test isapprox(qtt_eval(q, i, j, n, ord), A[i+1, j+1]; atol = 1e-10)
        end
        # error bound of the truncation
        for eps in (1e-1, 1e-2)
            q = ttsvd(to_chain(A, ord); eps = eps)
            B = qtt_to_array(q, n, ord)
            @test norm(A - B) <= eps * norm(A) * (1 + 1e-10)
        end
    end
    # smooth function: low rank in interleaved order
    n = 8; N = 2^n
    A = [sin(3 * (i + 0.5) / N) * exp(-(j + 0.5) / N) + cos(5 * (i + j + 1) / N) for i in 0:N-1, j in 0:N-1]
    for ord in (:interleaved, :sequential)
        q = ttsvd(to_chain(A, ord); eps = 1e-10)
        println("smooth test ", ord, " ranks ", qtt_ranks(q))
        @test maximum(qtt_ranks(q)) <= 8
    end
end

@testset "agreement with quantics.jl" begin
    for ord in (:interleaved, :sequential)
        n = 5; N = 2^n
        A = randn(N, N)
        g = QuanticsGrid(n, ord, (0.0, 1.0), (0.0, 1.0))
        @test all(g.pos[1] .== site_positions(n, ord)[1]) && all(g.pos[2] .== site_positions(n, ord)[2])
        q = ttsvd(to_chain(A, ord); eps = 1e-14)
        maxdev_mine = maximum(abs(qtt_eval(q, i, j, n, ord) - A[i+1, j+1]) for i in 0:N-1, j in 0:N-1)
        # my chain layout read with quantics.jl's idx_to_bits (site state 1/2 -> bit 0/1)
        devbits = maximum(begin
            s = idx_to_bits(g, (i, j))
            v = ones(1, 1)
            for p in 1:2n; v = v * q.cores[p][:, s[p], :]; end
            abs(v[1] - A[i+1, j+1])
        end for i in 0:N-1, j in 0:N-1)
        psi = array_to_mps(A, g; cutoff = 1e-14)
        E = MPSEvaluator(psi)
        dev_eval = maximum(abs(E(idx_to_bits(g, (i, j))) - A[i+1, j+1]) for i in 0:N-1, j in 0:N-1)
        dev_arr = maximum(abs.(mps_to_array(psi, g) - A))
        @printf("%s: mine %.1e  mine-with-idx_to_bits %.1e  array_to_mps+MPSEvaluator %.1e  array_to_mps+mps_to_array %.1e\n",
                ord, maxdev_mine, devbits, dev_eval, dev_arr)
        @test maxdev_mine < 1e-10
        @test devbits < 1e-10
        @test dev_arr < 1e-10
        dev_eval < 1e-10 || println("   NOTE: array_to_mps + MPSEvaluator disagree with idx_to_bits for ", ord)
        # ranks: ITensors cutoff = discarded weight / norm^2 per bond; compare with eps^2/(L-1)
        cut = 1e-4
        psi2 = array_to_mps(A, g; cutoff = cut)
        q2 = ttsvd(to_chain(A, ord); eps = sqrt(cut * (2n - 1)))
        println("   linkdims ITensors ", linkdims(psi2), "\n   ranks mine        ", qtt_ranks(q2))
    end
end

@testset "wavelets" begin
    N = 64; A = randn(N, N)
    for (f, fi) in ((haar_fwd!, haar_inv!), (cdf97_fwd!, cdf97_inv!))
        C = wt2d!(copy(A), 4, f)
        B = iwt2d!(copy(C), 4, fi)
        @test maximum(abs.(A - B)) < 1e-10
        @printf("%s energy ratio %.4f\n", f, norm(C) / norm(A))
    end
    # smooth field: CDF 9/7 compacts energy better than Haar
    S = [sin(2π * (i + 0.5) / N) * cos(3π * (j + 0.5) / N) for i in 0:N-1, j in 0:N-1]
    for f in (haar_fwd!, cdf97_fwd!)
        C = wt2d!(copy(S), 3, f)
        sa = sort(abs.(vec(C)), rev = true)
        @printf("%s: fraction of energy in top 1%% coefficients %.6f\n", f, sum(abs2, sa[1:41]) / sum(abs2, sa))
    end
end

@testset "zlib + fill" begin
    N = 256
    A = [100 * sin((i + j) / 30) for i in 1:N, j in 1:N]
    b, Q, T = quant_zlib(A, 0.1)
    @test maximum(abs.(A - Q)) <= 0.05 + 1e-9
    println("zlib bytes ", b, " (raw Float32 ", 4N^2, "), int type ", T)
    valid = [hypot(i - 128, j - 128) < 80 for i in 1:N, j in 1:N]
    F = harmonic_fill(A, valid)
    @test all(F[valid] .== A[valid])
    @test all(isfinite, F)
    println("fill range ", extrema(F[.!valid]))
end
println("all tests done")
