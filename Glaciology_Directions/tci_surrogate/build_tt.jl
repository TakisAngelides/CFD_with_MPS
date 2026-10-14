# Tensor cross interpolation (TCI2) of ice-sheet QoIs over d physical parameters.
#   f(i_1, ..., i_d, q) = QoI_q(x(i)) / QoI_q(nominal),  x_k(i_k) = Gauss-Legendre node i_k of parameter k,
# the last TT leg is the QoI index (all 10 QoIs come from one model run, so this leg is free).
# Model runs are cached by parameter multi-index and evaluated in parallel (threads) inside a TCI batch
# evaluator. One TCI per tolerance (fresh start, shared cache) -> cost (# distinct runs requested) vs accuracy.
#   sbatch -J tt6 slurm/run.sbatch build_tt.jl d=6 nnodes=9 nbits=5 tols=1e-2,1e-3,1e-4
using LinearAlgebra, Printf, Random, Serialization, Statistics
import TensorCrossInterpolation as TCI
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl"); include("tt_tools.jl")

cfg = Dict{String,Any}("d" => 6, "nnodes" => 9, "nbits" => 5, "tols" => "1e-2,1e-3,1e-4", "pivot" => "rook",
                       "nglobal" => 5, "maxbond" => 200, "maxiter" => 20, "ngrid" => 200, "tag" => "", "val" => "", "qsel" => "1,2,3,4,5,6,7,8,9,10")
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end
d = cfg["d"]; nn = cfg["nnodes"]; nb = cfg["nbits"]
ps = ParamSpace(d, nn)
base = "d$(d)_n$(nn)_b$(nb)"
cachefile = "output/cache_$(base).jls"
outfile = "output/tt_$(base)$(cfg["tag"]).jls"
mkpath("output")
println("TCI surrogate: d = $d ", PARAM_NAMES[ps.active], ", $nn GL nodes, model $(2^nb)^2, threads $(Threads.nthreads())")
println("cfg: ", cfg); flush(stdout)

# ---------------------------------------------------------------------------------------------- model runs
function robust_run(par::ModelParams, nbits)
    for (k, dt_tr) in enumerate((500.0, 250.0, 125.0))
        try
            q, info = run_model(par; nbits, dt_tr)
            all(isfinite, q) || error("non-finite QoI")
            return q, (; info..., dt_tr, retry = k - 1)
        catch e
            k == 3 && rethrow(e)
            @warn "run failed with dt_tr = $dt_tr, retrying" par exception = e
        end
    end
end

cache = isfile(cachefile) ? deserialize(cachefile) :
        Dict{Symbol,Any}(:q => Dict{Vector{Int},Vector{Float64}}(), :info => Dict{Vector{Int},Any}(), :qref => nothing)
println("cache: $(length(cache[:q])) runs from $cachefile")
if cache[:qref] === nothing
    run_model(model_params(ps, zeros(d)); nbits = nb)        # compile
    cache[:qref], _ = run_model(model_params(ps, zeros(d)); nbits = nb)
end
qsel = parse.(Int, split(cfg["qsel"], ","))
qref = cache[:qref][qsel]
println("QoIs: ", join([@sprintf("%s=%.4f", l, x) for (l, x) in zip(qoi_labels()[qsel], qref)], " "))
nq = length(qsel)

mutable struct ModelEvaluator <: TCI.BatchEvaluator{Float64}
    localdims::Vector{Int}
    requested::Set{Vector{Int}}
    nnew::Int
    teval::Float64
    tlast::Float64
end
const EV = ModelEvaluator(vcat(fill(nn, d), nq), Set{Vector{Int}}(), 0, 0.0, time())

function ensure!(ev::ModelEvaluator, keys)
    union!(ev.requested, keys)
    missing = unique([k for k in keys if !haskey(cache[:q], k)])
    isempty(missing) && return
    out = Vector{Any}(undef, length(missing))
    t = @elapsed Threads.@threads :greedy for j in eachindex(missing)
        out[j] = robust_run(model_params(ps, node_coords(ps, missing[j])), nb)
    end
    for j in eachindex(missing)
        cache[:q][missing[j]] = out[j][1]; cache[:info][missing[j]] = out[j][2]
    end
    ev.nnew += length(missing); ev.teval += t
    if time() - ev.tlast > 60
        ev.tlast = time()
        @printf("   [%s] runs: requested %d, cache %d, eval wall %.0f s\n", Libc.strftime("%H:%M:%S", time()), length(ev.requested),
                length(cache[:q]), ev.teval)
        flush(stdout)
    end
end

fval(idx) = cache[:q][idx[1:d]][qsel[idx[d+1]]] / qref[idx[d+1]]

function (ev::ModelEvaluator)(idx::Vector{Int})
    ensure!(ev, [idx[1:d]])
    return fval(idx)
end

function (ev::ModelEvaluator)(L::Vector{Vector{Int}}, R::Vector{Vector{Int}}, ::Val{M}) where {M}
    if length(L) * length(R) == 0
        return Array{Float64,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    nl = length(first(L))
    cind = vec(collect(Iterators.product(ntuple(i -> 1:ev.localdims[nl+i], M)...)))
    full = [vcat(l, collect(c), r) for l in L, c in cind, r in R]
    ensure!(ev, unique([x[1:d] for x in vec(full)]))
    res = [fval(x) for x in full]
    return reshape(res, length(L), ev.localdims[nl+1:nl+M]..., length(R))
end

# ---------------------------------------------------------------------------------------------- validation data
val = nothing
if cfg["val"] != "" && isfile(cfg["val"])
    val = deserialize(cfg["val"])
    println("validation set $(cfg["val"]): $(size(val[:U], 1)) points")
end
lam = bary_weights(ps.t)
function val_errors(C)
    val === nothing && return nothing
    P = reduce(hcat, [tt_eval_cont(C, ps.t, lam, val[:U][k, :]) for k in 1:size(val[:U], 1)])' .* qref'
    Qs = val[:Q][:, qsel]; E = P .- Qs
    return (; rel_max = vec(maximum(abs.(E ./ Qs); dims = 1)), rel_rms = vec(sqrt.(mean((E ./ Qs) .^ 2; dims = 1))),
            sd_rms = vec(sqrt.(mean(E .^ 2; dims = 1)) ./ std(Qs; dims = 1)))
end

# ---------------------------------------------------------------------------------------------- TCI runs
results = isfile(outfile) ? deserialize(outfile) : Dict{Float64,Any}()
tols = parse.(Float64, split(cfg["tols"], ","))
center = vcat(fill((nn + 1) ÷ 2, d), 1)
for tol in tols
    empty!(EV.requested); EV.nnew = 0; EV.teval = 0.0
    tw = @elapsed tci, ranks, errors = TCI.crossinterpolate2(Float64, EV, EV.localdims, [center]; tolerance = tol,
                                                             maxbonddim = cfg["maxbond"], maxiter = cfg["maxiter"],
                                                             pivotsearch = Symbol(cfg["pivot"]), nsearchglobalpivot = cfg["nglobal"],
                                                             maxnglobalpivot = min(cfg["nglobal"], 5), verbosity = 1, loginterval = 1)
    C = Array{Float64,3}.(TCI.sitetensors(TCI.TensorTrain(tci)))
    nreq = length(EV.requested)
    serialize(cachefile, cache)
    ve = val_errors(C)
    mu, vr, S1, ST = tt_stats(C, ps.w)
    results[tol] = Dict(:cores => C, :ranks => tt_ranks(C), :nreq => nreq, :nnew => EV.nnew, :wall => tw, :teval => EV.teval,
                        :iters => length(ranks), :pivot_errors => errors, :val => ve, :mean => mu .* qref,
                        :std => sqrt.(max.(vr, 0)) .* qref, :S1 => S1, :ST => ST, :requested => collect(EV.requested))
    @printf("\n== tol %.0e: %d iterations, ranks %s (max %d), TT params %d\n", tol, length(ranks), string(tt_ranks(C)),
            maximum(tt_ranks(C)), tt_nparams(C))
    @printf("   distinct model runs requested %d (new %d), full grid %.3e (ratio %.2e), wall %.0f s (model %.0f s)\n",
            nreq, EV.nnew, float(nn)^d, nreq / float(nn)^d, tw, EV.teval)
    @printf("   last pivot error %.2e\n", errors[end])
    if ve !== nothing
        println("   validation (continuous random points): rel max ", join([@sprintf("%.1e", x) for x in ve.rel_max], " "))
        println("                                          rel rms ", join([@sprintf("%.1e", x) for x in ve.rel_rms], " "))
        println("                                          rms/std ", join([@sprintf("%.1e", x) for x in ve.sd_rms], " "))
    end
    println("   TT mean ", join([@sprintf("%.4f", x) for x in mu .* qref], " "))
    println("   TT std  ", join([@sprintf("%.4f", x) for x in sqrt.(max.(vr, 0)) .* qref], " "))
    flush(stdout)
    serialize(cachefile, cache); serialize(outfile, results)
end

# ---------------------------------------------------------------------------------------------- grid error
if cfg["ngrid"] > 0
    rng = MersenneTwister(12345)
    gpts = [rand(rng, 1:nn, d) for _ in 1:cfg["ngrid"]]
    ncached = count(k -> haskey(cache[:q], k), gpts)
    ensure!(EV, gpts)
    println("\n== error on $(length(gpts)) random grid points ($ncached were already in the cache)")
    for tol in sort(collect(keys(results)); rev = true)
        C = results[tol][:cores]
        E = reduce(hcat, [tt_eval_params(C, k) .- cache[:q][k][qsel] ./ qref for k in gpts])'
        results[tol][:grid_err] = vec(maximum(abs.(E); dims = 1))
        @printf("   tol %.0e: max |err| / nominal per QoI %s\n", tol, join([@sprintf("%.1e", x) for x in results[tol][:grid_err]], " "))
    end
    serialize(cachefile, cache); serialize(outfile, results)
end
nsplit = count(i -> i.nsplit > 0, values(cache[:info])); nretry = count(i -> i.retry > 0, values(cache[:info]))
nnc = count(i -> !i.converged, values(cache[:info]))
println("\ncache: $(length(cache[:q])) runs; runs with split steps $nsplit, retried with smaller dt $nretry, steady not converged $nnc")
println("mean Newton iterations per run ", mean(i.nnewton for i in values(cache[:info])))
