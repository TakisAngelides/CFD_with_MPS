# Accuracy of the assembled Newton matrix J = I - dt dF/dH (1D Halfar, dt = 10 a) against the
# sparse Jacobian of the same discrete scheme, as a function of the operator cutoff and the MPO
# product algorithm: action J v on a consistent v, and the Newton update J dH = -R solved with
# the direct local solver vs a sparse direct solve (same right-hand side).
# Fields use the exact product with cutoff 1e-24 (see bench_precision.jl).
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_jacobian.jl bits=10,12,14
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "10,12,14", "dt" => "10", "ocs" => "1e-15,1e-20,1e-25,1e-30", "algs" => "zipup,naive")
for a in ARGS
    k, v = split(a, "="); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
dt = parse(Float64, opts["dt"])

p = SIAParams()
hb = Halfar(d = 1, p = p)
G = Gamma(p)
t0 = halfar_t0(hb)
PRODUCT_ALG[] = "naive"

for nbits in bits
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3))
    ops = SIAOperators(g)
    stf = SolverSettings(cutoff = 1e-24)
    H0 = function_to_mps(x -> halfar_H(hb, t0, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    H1 = function_to_mps(x -> halfar_H(hb, t0 + dt, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    F1 = face_fields(H1, nothing, ops, p, stf)
    R = lincomb([1.0, -dt, -1.0], [H1, flux_divergence(F1, ops, stf), H0]; cutoff = 1e-24)
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    h0 = mps_to_array(H0, g); h1 = mps_to_array(H1, g)
    He = Avg * h1; gg = Df * h1
    Js = I - dt * Db * (spdiagm(0 => 3G .* He .^ 5 .* gg .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* gg .^ 3) * Avg)
    Rs = h1 - dt * Db * (G .* He .^ 5 .* gg .^ 3) - h0
    rhs = mps_to_array(R, g)
    xs = Js \ (-rhs)
    v = (1 / maximum(h1)) * H1; vd = h1 / maximum(h1)
    Jv = Js * vd
    println(@sprintf("\n2^%d points: |R(MPS) - R(sparse)|/|R| %.1e, chi(R) %d", nbits, norm(rhs - Rs) / norm(Rs), maxlinkdim(R)))
    for malg in split(opts["algs"], ","), oc in parse.(Float64, split(opts["ocs"], ","))
        MPO_PRODUCT_ALG[] = malg
        st = SolverSettings(cutoff = 1e-16, op_cutoff = oc, op_maxdim = 512, lin_cutoff = 1e-10, lin_maxdim = 32, nsweeps = 1)
        tJ = @elapsed J = mpo_lincomb([1.0, -dt], [ops.I, jacobian_mpo(F1, ops, p, st)]; cutoff = oc, maxdim = 512)
        Jvm = mps_to_array(mps_apply(J, v; cutoff = 1e-28, maxdim = 1024), g)
        ts = @elapsed x, (res, tries) = _linsolve(J, -1.0 * R, -1.0 * R, st, false)
        println(@sprintf("   %-5s op_cutoff %.0e: chi(J) %3d (%.2f s)  Jv rel err %.1e | solve %.2f s chi(dH) %3d res %.1e (%d retries)  dH rel err vs sparse %.1e",
                         malg, oc, maxlinkdim(J), tJ, norm(Jvm - Jv) / norm(Jv), ts, maxlinkdim(x), res, tries,
                         norm(mps_to_array(x, g) - xs) / norm(xs)))
        flush(stdout)
    end
end
