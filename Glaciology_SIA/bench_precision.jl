# Precision of the MPS SIA flux F(H) = Db (D Df H) and of the Jacobian action against the dense
# arrays of the same discrete scheme (1D Halfar profile), for the MPO-MPS product algorithm and
# the field cutoff. The margin cell is where truncation errors are amplified.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_precision.jl bits=10,12,14
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "10,12,14", "dt" => "10", "algs" => "densitymatrix,naive", "cuts" => "1e-13,1e-18,1e-24")
for a in ARGS
    k, v = split(a, "="); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
dt = parse(Float64, opts["dt"])

p = SIAParams()
hb = Halfar(d = 1, p = p)
G = Gamma(p)
t1 = halfar_t0(hb) + dt

for nbits in bits
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3))
    ops = SIAOperators(g)
    H = function_to_mps(x -> halfar_H(hb, t1, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    hd = mps_to_array(H, g)
    He = Avg * hd; gg = Df * hd
    Fd = Db * (G .* He .^ 5 .* gg .^ 3)
    Js = Db * (spdiagm(0 => 3G .* He .^ 5 .* gg .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* gg .^ 3) * Avg)
    v = hd / maximum(hd)                       # test vector for the Jacobian action
    Jv = Js * v
    println(@sprintf("\n2^%d points, chi(H) %d, max|dt F| %.1f m", nbits, maxlinkdim(H), dt * maximum(abs, Fd)))
    for alg in split(opts["algs"], ","), cut in parse.(Float64, split(opts["cuts"], ","))
        PRODUCT_ALG[] = alg
        st = SolverSettings(prod_cutoff = cut)
        tF = @elapsed begin
            F = face_fields(H, nothing, ops, p, st)
            Fm = mps_to_array(flux_divergence(F, ops, st), g)
        end
        tJ = @elapsed J = jacobian_mpo(F, ops, p, st)
        Jvm = mps_to_array(mps_apply(J, array_to_mps(v, g; cutoff = 1e-16); cutoff = 1e-16, maxdim = 512), g)
        e = abs.(Fm - Fd); i = argmax(e)
        println(@sprintf("   %-13s cutoff %.0e: max|dt dF| %.2e m at x = %7.1f km  rel %.1e  chi(D) %3d  %.2f s | J v rel err %.1e  chi(J) %3d  %.2f s",
                         alg, cut, dt * e[i], coord(g, 1, i - 1) / 1e3, norm(Fm - Fd) / norm(Fd), maxlinkdim(F[1].D), tF,
                         norm(Jvm - Jv) / norm(Jv), maxlinkdim(J), tJ))
        flush(stdout)
    end
end
