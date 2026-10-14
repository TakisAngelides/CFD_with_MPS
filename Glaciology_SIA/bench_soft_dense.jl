# Intrinsic QTT ranks of H and of the SIA face diffusivity D (Cartesian 2D, interleaved), computed
# from exact dense arrays and compressed by SVD at several cutoffs -- independent of TCI noise and
# of MPS-arithmetic truncation. Cases: sharp Halfar dome ("radial") and softplus-smoothed margins of
# fixed physical width w_km (see bench_polar_ranks.jl). Does chi saturate once the grid resolves w?
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_soft_dense.jl bits=6,7,8,9,10 w_km=10,50
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

opts = Dict("bits" => "6,7,8,9,10", "w_km" => "10,50", "cutoffs" => "1e-8,1e-12,1e-16", "eps" => "0.0")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
cutoffs = parse.(Float64, split(opts["cutoffs"], ","))
epsw = parse(Float64, opts["eps"])
p = SIAParams(); hb = Halfar(d = 2, p = p); t0 = halfar_t0(hb)
Rdom = 1200e3; G = Gamma(p); n = p.n_glen

function soft_H(r, w)
    inner = 1 - (r / hb.R0)^((n + 1) / n)
    w == 0 && return hb.H0 * max(inner, 0.0)^(n / (2n + 1))
    sp = inner / w > 40 ? inner : w * log1p(exp(inner / w))
    return hb.H0 * sp^(n / (2n + 1))
end

# x-face diffusivity (Mahaffy type II), zero ghosts
function dense_Dx(H, h)
    N = size(H, 1)
    P = zeros(N + 2, N + 2); P[2:N+1, 2:N+1] .= H
    D = zeros(N, N)
    for j in 1:N, i in 1:N
        I, J = i + 1, j + 1
        Hf = (P[I, J] + P[I+1, J]) / 2
        sx = (P[I+1, J] - P[I, J]) / h
        sy = (P[I, J+1] - P[I, J-1] + P[I+1, J+1] - P[I+1, J-1]) / (4h)
        D[i, j] = G * Hf^(n + 2) * (sx^2 + sy^2)^div(n - 1, 2)
    end
    return D
end

cases = [("radial", 0.0); [("soft$(w)", parse(Float64, w) * 1e3 / (0.75 * hb.R0)) for w in split(opts["w_km"], ",")]]
@printf("%-9s %4s %6s %8s", "case", "bits", "h[km]", "w/h")
for c in cutoffs
    @printf("  chiH(%.0e) chiD(%.0e)", c, c)
end
println(); flush(stdout)
for nb in bits, (name, w) in cases
    g = QuanticsGrid(nb, :interleaved, (-Rdom, Rdom), (-Rdom, Rdom))
    N = 2^nb; h = 2Rdom / N
    x = [-Rdom + (i - 0.5) * h for i in 1:N]
    H = [soft_H(hypot(xi, yj) / (1 + epsw * cos(3 * atan(yj, xi))), w) for xi in x, yj in x]
    D = dense_Dx(H, h)
    @printf("%-9s %4d %6.2f %8.2f", name, nb, h / 1e3, w * 0.75 * hb.R0 / h)
    for c in cutoffs
        @printf("  %10d %10d", maxlinkdim(array_to_mps(H, g; cutoff = c)), maxlinkdim(array_to_mps(D, g; cutoff = c)))
    end
    println(); flush(stdout)
end
