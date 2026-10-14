# Robustness of the transient time grid (step splits) and of the steady-state phase over the parameter box.
#   sbatch -J rob5 slurm/run.sbatch check_robust.jl nbits=5
using LinearAlgebra, Printf, Random, Statistics
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl")
cfg = Dict{String,Any}("nbits" => 5, "nrand" => 160, "nss" => 32, "verbose_pt" => 0)
for arg in ARGS
    k, v = split(arg, "=", limit = 2); cfg[k] = parse(typeof(cfg[k]), v)
end
nb = cfg["nbits"]; ps = ParamSpace(8, 9)
rng = MersenneTwister(2)
pts = vcat([[((c >> (k - 1)) & 1 == 1 ? 1.0 : -1.0) for k in 1:8] for c in 0:255], [2 .* rand(rng, 8) .- 1 for _ in 1:cfg["nrand"]])
run_model(model_params(ps, zeros(8)); nbits = nb)   # compile
function sweep(f, pts)
    out = Vector{Any}(undef, length(pts))
    t = @elapsed Threads.@threads :greedy for k in eachindex(pts)
        out[k] = try
            t0 = time(); q, info = f(pts[k]); (q, info, time() - t0)
        catch e
            (e, nothing, 0.0)
        end
    end
    return out, t
end
for (name, kw) in [("fixed500", (; dt_tr = 500.0, graded = false))]
    out, t = sweep(u -> run_model(model_params(ps, u); nbits = nb, ss = false, kw...), pts)
    nfail = count(o -> o[2] === nothing, out)
    ok = [o for o in out if o[2] !== nothing]
    nspl = [o[2].nsplit for o in ok]
    tspl = reduce(vcat, [o[2].tsplit for o in ok]; init = Float64[])
    @printf("transient %-10s: %d runs, %.1f s wall, mean run %.2f s, failed %d, runs with splits %d (max %d), split times: %s\n",
            name, length(pts), t, mean(o[3] for o in ok), nfail, count(>(0), nspl), maximum(nspl),
            isempty(tspl) ? "-" : string(sort(unique(tspl))[1:min(end, 12)]))
    for (k, o) in enumerate(out)
        o[2] === nothing && println("   FAILED at ", pts[k], ": ", o[1])
        o[2] !== nothing && o[2].nsplit > 0 && k <= 300 && println("   split ", o[2].nsplit, " at ", join([@sprintf("%+.2f", x) for x in pts[k]], " "))
    end
end
# steady phase variants on random points
sp = pts[257:256+cfg["nss"]]
res = Dict{String,Any}()
for (name, kw) in [("rate1e-12", (; dt_max = 1e6, ss_change_tol = 1e-6, ss_growth = 2.0)), ("polish_g2", (; dt_max = 1e10, ss_change_tol = 1e-8, ss_growth = 2.0)),
                   ("polish_g4", (; dt_max = 1e10, ss_change_tol = 1e-8, ss_growth = 4.0))]
    out, t = sweep(u -> run_model(model_params(ps, u); nbits = nb, kw...), sp)
    ok = [o for o in out if o[2] !== nothing]
    @printf("steady %-10s: %.1f s wall, mean run %.2f s, failed %d, not converged %d, mean nss %.1f, max nss %d, mean newton %.0f, rej %d\n",
            name, t, mean(o[3] for o in ok), length(out) - length(ok), count(o -> !o[2].converged, ok), mean(o[2].nss for o in ok),
            maximum(o[2].nss for o in ok), mean(o[2].nnewton for o in ok), sum(o[2].nrej for o in ok))
    res[name] = out
end
for a in ["rate1e-12", "polish_g4"]
    d = [maximum(abs.(res[a][k][1][6:10] ./ res["polish_g2"][k][1][6:10] .- 1)) for k in eachindex(sp) if res[a][k][2] !== nothing]
    @printf("max rel. difference of steady QoIs, %s vs polish_g2: max %.2e, median %.2e\n", a, maximum(d), median(d))
end
if cfg["verbose_pt"] > 0
    run_model(model_params(ps, sp[1]); nbits = nb, verbose = true)
end
