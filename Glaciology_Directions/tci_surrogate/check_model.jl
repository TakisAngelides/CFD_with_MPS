# Checks of the callable model: (1) reproduces Glaciology_SIA/dense_landau_2d.jl, (2) run time per model
# evaluation, (3) robustness at corners of the parameter box, (4) smoothness / noise of the QoIs.
#   sbatch slurm/run.sbatch check_model.jl nbits=5
using LinearAlgebra, Printf, Random
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl")
cfg = Dict{String,Any}("nbits" => 5, "replica" => 1, "ncorner" => 32, "fd" => 1)
for arg in ARGS
    k, v = split(arg, "=", limit = 2); cfg[k] = parse(typeof(cfg[k]), v)
end
nb = cfg["nbits"]
println("threads: ", Threads.nthreads())
if cfg["replica"] == 1
    println("== replica of dense_landau_2d.jl case=ellipse nbits=5 dt=1000 t_run=150000 steady_tol=1e-9")
    println("   (original log L2ellipse5: 25 steps, t = 25000 a: margin 490.011 .. 654.073 km, h(pole) 2874.05 m)")
    run_replica(ModelParams(ell = 0.2); nbits = 5)
    println("== replica case=eismint (original L2eismint5: 25 steps, t = 25000 a: margin 580.017 .. 580.017 km, h(pole) 2979.23 m)")
    run_replica(ModelParams(); nbits = 5, eismint = true)
end
ps = ParamSpace(8, 9)
println("== nominal run, nbits = $nb")
run_model(model_params(ps, zeros(8)); nbits = nb)   # compile
tm = @elapsed q, info = run_model(model_params(ps, zeros(8)); nbits = nb, verbose = true)
@printf("nominal: %.2f s, info %s\n", tm, string(info))
println("QoIs ", join(qoi_labels(), " "))
println("     ", join([@sprintf("%.6f", x) for x in q], " "))
# corners of the 8-d box (random subset), in parallel
rng = MersenneTwister(1)
corners = [rand(rng, [-1.0, 1.0], 8) for _ in 1:cfg["ncorner"]]
push!(corners, [1.0, 1, 1, 1, 1, 1, -1, 1]); push!(corners, [-1.0, -1, -1, -1, 1, 1, 1, -1])
push!(corners, [1.0, -1, -1, -1, 1, 1, -1, 1]); push!(corners, [-1.0, 1, 1, 1, 1, 1, 1, -1])
res = Vector{Any}(undef, length(corners))
tc = @elapsed Threads.@threads :greedy for k in eachindex(corners)
    t0 = time()
    try
        q, info = run_model(model_params(ps, corners[k]); nbits = nb)
        res[k] = (q, info, time() - t0)
    catch e
        res[k] = (e, nothing, time() - t0)
    end
end
@printf("== %d corners in %.1f s wall\n", length(corners), tc)
for k in eachindex(corners)
    q, info, tk = res[k]
    if info === nothing
        println("corner ", corners[k], " FAILED: ", q)
    else
        @printf("%s  %5.1f s  newton %4d split %d rej %d nss %3d conv %d | %s\n", join([c > 0 ? "+" : "-" for c in corners[k]]), tk,
                info.nnewton, info.nsplit, info.nrej, info.nss, info.converged, join([@sprintf("%.4f", x) for x in q], " "))
    end
end
if cfg["fd"] == 1
    println("== smoothness: central differences d q / d u_k at the nominal point, steps 1e-1 .. 1e-5")
    for k in 1:8
        steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ders = Vector{Vector{Float64}}(undef, length(steps))
        Threads.@threads for s in eachindex(steps)
            e = zeros(8); e[k] = steps[s]
            qp, _ = run_model(model_params(ps, e); nbits = nb); qm, _ = run_model(model_params(ps, -e); nbits = nb)
            ders[s] = (qp - qm) / (2steps[s])
        end
        println("param $(PARAM_NAMES[k]):")
        for s in eachindex(steps)
            println(@sprintf("   h=%.0e ", steps[s]), join([@sprintf("% .6e", x) for x in ders[s]], " "))
        end
    end
end
