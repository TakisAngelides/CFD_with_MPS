# Held-out model runs at random (continuous) parameter points, laid out as a Saltelli pick-freeze design
# (rows of A, B, AB_1..AB_d; every row is uniform on the box), so the same runs serve as validation set for
# the TT surrogate and as an honest Monte Carlo baseline (moments and Sobol indices from real runs).
#   sbatch -J val6 slurm/run.sbatch validate.jl d=6 nbits=5 N=100
using LinearAlgebra, Printf, Random, Serialization, Statistics
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl"); include("tt_tools.jl")
cfg = Dict{String,Any}("d" => 6, "nbits" => 5, "N" => 100, "seed" => 7)
for arg in ARGS
    k, v = split(arg, "=", limit = 2); cfg[k] = parse(typeof(cfg[k]), v)
end
d = cfg["d"]; nb = cfg["nbits"]; N = cfg["N"]
ps = ParamSpace(d, 9)
A, B, AB = saltelli_design(MersenneTwister(cfg["seed"]), N, d)
U = vcat(A, B, AB...)
npts = size(U, 1)
println("validation / MC design: d = $d, N = $N, $(npts) runs at $(2^nb)^2, threads $(Threads.nthreads())"); flush(stdout)
run_model(model_params(ps, zeros(d)); nbits = nb)   # compile
Q = fill(NaN, npts, 10); infos = Vector{Any}(undef, npts); tr = zeros(npts)
done = Threads.Atomic{Int}(0); t0 = time()
wall = @elapsed Threads.@threads :greedy for k in 1:npts
    t1 = time()
    for dt_tr in (500.0, 250.0, 125.0)
        try
            q, info = run_model(model_params(ps, U[k, :]); nbits = nb, dt_tr)
            Q[k, :] = q; infos[k] = (; info..., dt_tr); break
        catch e
            @warn "run $k failed with dt_tr = $dt_tr" exception = e
        end
    end
    tr[k] = time() - t1
    n = Threads.atomic_add!(done, 1) + 1
    n % 100 == 0 && (@printf("  %d / %d runs, %.0f s\n", n, npts, time() - t0); flush(stdout))
end
nfail = count(k -> any(isnan, Q[k, :]), 1:npts)
@printf("%d runs in %.0f s wall (mean %.2f s per run), failed %d, with split steps %d, steady not converged %d\n", npts, wall,
        mean(tr), nfail, count(k -> isassigned(infos, k) && infos[k].nsplit > 0, 1:npts),
        count(k -> isassigned(infos, k) && !infos[k].converged, 1:npts))
mkpath("output")
fn = "output/val_d$(d)_b$(nb)_N$(N)_s$(cfg["seed"]).jls"
serialize(fn, Dict(:U => U, :Q => Q, :N => N, :d => d, :nbits => nb, :info => infos, :times => tr))
println("saved $fn")
println("MC (all $npts rows) mean ", join([@sprintf("%.4f", x) for x in mean(Q; dims = 1)], " "))
println("MC (all $npts rows) std  ", join([@sprintf("%.4f", x) for x in std(Q; dims = 1)], " "))
