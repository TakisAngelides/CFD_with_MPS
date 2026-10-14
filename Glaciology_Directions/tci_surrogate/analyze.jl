# Analysis of the TCI surrogates of one configuration (d, nnodes, nbits):
#   1. cost vs accuracy (distinct model runs, ranks, validation error on held-out random points),
#   2. moments and Sobol indices from the TT (exact GL quadrature) vs Monte Carlo from real runs,
#   3. equal-budget comparison: TT built with n runs vs MC / pick-freeze with n runs (MC sampling error
#      emulated by repeating the MC estimators on the validated reference surrogate).
#   sbatch -J an6 slurm/run.sbatch analyze.jl d=6 val=output/val_d6_b5_N100_s7.jls
using LinearAlgebra, Printf, Random, Serialization, Statistics
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl"); include("tt_tools.jl")
cfg = Dict{String,Any}("d" => 6, "nnodes" => 9, "nbits" => 5, "val" => "", "tag" => "", "reps" => 300, "ref_tol" => 0.0,
                       "seed" => 99, "qsel" => "1,2,3,4,5,6,7,8,9,10")
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end
d = cfg["d"]; nn = cfg["nnodes"]; nb = cfg["nbits"]
ps = ParamSpace(d, nn); lam = bary_weights(ps.t)
base = "d$(d)_n$(nn)_b$(nb)"
res = deserialize("output/tt_$(base)$(cfg["tag"]).jls")
qsel = parse.(Int, split(cfg["qsel"], ","))
qref = deserialize("output/cache_$(base).jls")[:qref][qsel]
labels = qoi_labels()[qsel]; nq = length(labels)
tols = sort(collect(keys(res)); rev = true)
reftol = cfg["ref_tol"] > 0 ? cfg["ref_tol"] : minimum(tols)
ref = res[reftol]; Cref = ref[:cores]
println("config $base, tolerances ", tols, ", reference TT tol $reftol")
fmt(v; f = "%.2e") = join([Printf.format(Printf.Format(f), x) for x in v], " ")

val = cfg["val"] == "" ? nothing : deserialize(cfg["val"])
Uv = val === nothing ? nothing : val[:U]; Qv = val === nothing ? nothing : val[:Q][:, qsel]
summary = Dict{Symbol,Any}(:d => d, :nn => nn, :nb => nb, :labels => labels, :tols => tols, :reftol => reftol)

# ------------------------------------------------------------------ 1. cost vs accuracy
println("\n== cost vs accuracy (validation on $(val === nothing ? 0 : size(Uv, 1)) held-out random points)")
println(" tol     runs   grid=9^d     maxrank  TTparams  err_rms/std(10ka) err_rms/std(ss)  relmax(10ka) relmax(ss)  grid_err")
rows = []
for tol in tols
    r = res[tol]; C = r[:cores]
    if val !== nothing
        P = reduce(hcat, [tt_eval_cont(C, ps.t, lam, Uv[k, :]) for k in 1:size(Uv, 1)])' .* qref'
        E = P .- Qv
        sd = vec(sqrt.(mean(E .^ 2; dims = 1)) ./ std(Qv; dims = 1)); rm = vec(maximum(abs.(E ./ Qv); dims = 1))
    else
        sd = fill(NaN, nq); rm = fill(NaN, nq)
    end
    ge = haskey(r, :grid_err) ? maximum(r[:grid_err]) : NaN
    row = (; tol, runs = r[:nreq], maxrank = maximum(r[:ranks]), npar = tt_nparams(C), sd, rm, ge, ranks = r[:ranks])
    push!(rows, row)
    @printf(" %.0e %7d  %.2e  %4d  %8d       %.1e           %.1e          %.1e      %.1e     %.1e\n", tol, r[:nreq], float(nn)^d,
            row.maxrank, row.npar, maximum(sd), maximum(sd), maximum(rm), maximum(rm), ge)
end
summary[:rows] = rows
println("per-QoI rms/std at the reference tol: ", join([@sprintf("%s %.1e", l, x) for (l, x) in zip(labels, rows[end].sd)], ", "))

# ------------------------------------------------------------------ 2. statistics: TT vs MC (real runs)
mu, vr, S1, ST = tt_stats(Cref, ps.w)
mu_tt = mu .* qref; sd_tt = sqrt.(vr) .* qref
summary[:mu_tt] = mu_tt; summary[:sd_tt] = sd_tt; summary[:S1_tt] = S1; summary[:ST_tt] = ST
println("\n== moments: TT (tol $reftol, $(ref[:nreq]) runs) vs MC on the $(val === nothing ? 0 : size(Qv, 1)) validation runs")
if val !== nothing
    n = size(Qv, 1)
    mu_mc = vec(mean(Qv; dims = 1)); sd_mc = vec(std(Qv; dims = 1))
    summary[:mu_mc] = mu_mc; summary[:sd_mc] = sd_mc; summary[:n_mc] = n
end
for q in 1:nq
    if val !== nothing
        @printf("  %-12s mean TT %10.4f  MC %10.4f +- %.4f   std TT %9.4f  MC %9.4f\n", labels[q], mu_tt[q], summary[:mu_mc][q],
                summary[:sd_mc][q] / sqrt(summary[:n_mc]), sd_tt[q], summary[:sd_mc][q])
    else
        @printf("  %-12s mean TT %10.4f  std TT %9.4f\n", labels[q], mu_tt[q], sd_tt[q])
    end
end
names = string.(PARAM_NAMES[ps.active])
if val !== nothing
    N = val[:N]
    fA = Qv[1:N, :]; fB = Qv[N+1:2N, :]; fAB = [Qv[(2+k-1)*N+1:(2+k)*N, :] for k in 1:d]
    S1mc, STmc = pickfreeze(fA, fB, fAB)
    # bootstrap standard errors
    rng = MersenneTwister(1); nboot = 200
    bs = [begin ii = rand(rng, 1:N, N); pickfreeze(fA[ii, :], fB[ii, :], [x[ii, :] for x in fAB])[1] end for _ in 1:nboot]
    S1se = std(cat(bs...; dims = 3); dims = 3)[:, :, 1]
    summary[:S1_mc] = S1mc; summary[:ST_mc] = STmc; summary[:S1_mc_se] = S1se
end
println("\n== first-order Sobol indices S1 (TT exact quadrature", val === nothing ? ")" : " | MC pick-freeze with $(size(Qv, 1)) real runs +- bootstrap se)")
for q in 1:nq
    @printf("  %-12s TT: %s  (sum %.3f)\n", labels[q], join([@sprintf("%s %.3f", names[k], S1[k, q]) for k in 1:d], ", "), sum(S1[:, q]))
    val !== nothing && @printf("  %-12s MC: %s\n", "", join([@sprintf("%s %.3f+-%.3f", names[k], summary[:S1_mc][k, q], summary[:S1_mc_se][k, q]) for k in 1:d], ", "))
end
println("\n== total Sobol indices ST (TT)")
for q in 1:nq
    @printf("  %-12s TT: %s\n", labels[q], join([@sprintf("%s %.3f", names[k], ST[k, q]) for k in 1:d], ", "))
end

# ------------------------------------------------------------------ 3. equal budget: TT(tol) vs MC with the same runs
println("\n== equal budget: errors w.r.t. the reference TT, normalised by the QoI std (max over QoIs)")
println("   MC numbers: RMS over $(cfg["reps"]) repetitions emulated on the reference surrogate")
println(" tol    runs | TT: mean     std      S1(maxabs) | MC: mean     std      S1 pick-freeze(maxabs, N=runs/(d+2))")
rng = MersenneTwister(cfg["seed"])
surrogate(U) = reduce(hcat, [tt_eval_cont(Cref, ps.t, lam, U[k, :]) for k in 1:size(U, 1)])' .* qref'
eq = []
for tol in tols
    tol == reftol && continue
    r = res[tol]; n = r[:nreq]
    m2, v2, S12, _ = tt_stats(r[:cores], ps.w)
    e_mu = maximum(abs.(m2 .* qref .- mu_tt) ./ sd_tt); e_sd = maximum(abs.(sqrt.(v2) .* qref .- sd_tt) ./ sd_tt)
    e_s1 = maximum(abs.(S12 .- S1))
    # MC emulation
    reps = cfg["reps"]
    emu = zeros(reps, nq); esd = zeros(reps, nq); es1 = zeros(reps)
    Nb = max(n ÷ (d + 2), 2)
    seeds = rand(rng, UInt, reps)
    Threads.@threads for j in 1:reps
        rl = MersenneTwister(seeds[j])
        Q = surrogate(2 .* rand(rl, n, d) .- 1)
        emu[j, :] = (vec(mean(Q; dims = 1)) .- mu_tt) ./ sd_tt
        esd[j, :] = (vec(std(Q; dims = 1)) .- sd_tt) ./ sd_tt
        A, B, AB = saltelli_design(rl, Nb, d)
        s1, _ = pickfreeze(surrogate(A), surrogate(B), [surrogate(x) for x in AB])
        es1[j] = maximum(abs.(s1 .- S1))
    end
    mc_mu = maximum(sqrt.(mean(emu .^ 2; dims = 1))); mc_sd = maximum(sqrt.(mean(esd .^ 2; dims = 1))); mc_s1 = sqrt(mean(es1 .^ 2))
    push!(eq, (; tol, n, e_mu, e_sd, e_s1, mc_mu, mc_sd, mc_s1))
    @printf(" %.0e %6d |     %.1e  %.1e  %.1e    |     %.1e  %.1e  %.1e\n", tol, n, e_mu, e_sd, e_s1, mc_mu, mc_sd, mc_s1)
end
summary[:equal_budget] = eq
mkpath("output")
serialize("output/summary_$(base)$(cfg["tag"]).jls", summary)
println("\nsaved output/summary_$(base)$(cfg["tag"]).jls")
