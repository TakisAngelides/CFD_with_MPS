# Steady state by Newton on the non-degenerate steady equations vs backward-Euler polishing (dt -> 1e10).
#   sbatch -J chss slurm/run.sbatch check_steady.jl nbits=5
using LinearAlgebra, Printf, Random, Statistics
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl")
cfg = Dict{String,Any}("nbits" => 5, "npts" => 64)
for arg in ARGS
    k, v = split(arg, "=", limit = 2); cfg[k] = parse(typeof(cfg[k]), v)
end
nb = cfg["nbits"]; ps = ParamSpace(8, 9)
run_model(model_params(ps, zeros(8)); nbits = nb)
t1 = @elapsed q1, i1 = run_model(model_params(ps, zeros(8)); nbits = nb, verbose = true)
println("nominal (steady Newton): $t1 s ", i1)
t2 = @elapsed q2, i2 = run_model(model_params(ps, zeros(8)); nbits = nb, ss_newton = false)
println("nominal (BE polish):     $t2 s ", i2)
println("rel diff ", join([@sprintf("%.1e", x) for x in abs.(q1 ./ q2 .- 1)], " "))
rng = MersenneTwister(5)
pts = vcat([rand(rng, [-1.0, 1.0], 8) for _ in 1:cfg["npts"]÷2], [2 .* rand(rng, 8) .- 1 for _ in 1:cfg["npts"]÷2])
res = Vector{Any}(undef, length(pts))
tw = @elapsed Threads.@threads :greedy for k in eachindex(pts)
    a = @elapsed qa, ia = run_model(model_params(ps, pts[k]); nbits = nb)
    b = @elapsed qb, ib = run_model(model_params(ps, pts[k]); nbits = nb, ss_newton = false)
    res[k] = (qa, ia, a, qb, ib, b)
end
@printf("%d points, %.0f s wall\n", length(pts), tw)
d = [maximum(abs.(r[1][6:10] ./ r[4][6:10] .- 1)) for r in res]
@printf("steady Newton used in %d / %d runs; max rel diff of steady QoIs vs BE polish %.2e (median %.2e); transient QoIs identical: %s\n",
        count(r -> r[2].newton_used, res), length(res), maximum(d), median(d), all(r -> r[1][1:5] == r[4][1:5], res))
@printf("mean Newton iterations: steady-Newton runs %.0f, BE-polish runs %.0f; mean time %.2f vs %.2f s\n",
        mean(r[2].nnewton for r in res), mean(r[5].nnewton for r in res), mean(r[3] for r in res), mean(r[6] for r in res))
for (k, r) in enumerate(res)
    r[2].newton_used || println("  no steady Newton at ", pts[k], " ", r[2])
end
