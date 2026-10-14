using LinearAlgebra, Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")
p = SIAParams(); hb = Halfar(d = 1, p = p)
g = QuanticsGrid(8, :interleaved, (-1200e3, 1200e3)); ops = SIAOperators(g)
H = function_to_mps(x -> halfar_H(hb, halfar_t0(hb), abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
st = SolverSettings(direct_maxdim = 10)
F = face_fields(H, nothing, ops, p, st)
A = truncate!(ops.I - 10.0 * diffusion_mpo([f.D for f in F], ops, st); cutoff = 1e-15)
try
    x, r = _linsolve(A, H, H, st, true); println("spd ok res ", r)
    J = truncate!(ops.I - 10.0 * jacobian_mpo(F, ops, p, st); cutoff = 1e-15)
    x, r = _linsolve(J, H, H, st, false); println("nonsym ok res ", r)
catch e
    showerror(stdout, e, catch_backtrace())
end
