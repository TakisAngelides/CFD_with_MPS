# verbose trace of one model run (steady-phase diagnostics)
using LinearAlgebra, Printf
LinearAlgebra.BLAS.set_num_threads(1)
include("model.jl"); include("params.jl")
cfg = Dict{String,Any}("nbits" => 5, "growth" => 4.0, "tolss" => 1e-8, "dtmax" => 1e10, "tol" => 1e-9)
for arg in ARGS
    k, v = split(arg, "=", limit = 2); cfg[k] = parse(typeof(cfg[k]), v)
end
ps = ParamSpace(8, 9)
run_model(model_params(ps, zeros(8)); nbits = cfg["nbits"], ss = false)
t = @elapsed q, info = run_model(model_params(ps, zeros(8)); nbits = cfg["nbits"], ss_growth = cfg["growth"], ss_change_tol = cfg["tolss"],
                                 dt_max = cfg["dtmax"], verbose = true)
println(info, "  ", t, " s")
println(q)
