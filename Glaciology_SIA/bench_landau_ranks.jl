# Front-fixing (Landau) coordinates for 2D ice sheets: xi = r / R(theta), so the margin is the
# coordinate line xi = 1 for any star-shaped margin R(theta). Dense H(xi, theta) and the diffusivity
# D = Gamma H^5 |grad s|^2 (metric terms of the transform included) are compressed by SVD on an
# interleaved quantics grid and compared with the Cartesian ranks of the same ice sheet.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_landau_ranks.jl bits=6,7,8,9,10
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

opts = Dict("bits" => "6,7,8,9,10", "cutoff" => "1e-12")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
p = SIAParams(); hb = Halfar(d = 2, p = p); t0 = halfar_t0(hb); G = Gamma(p); n = p.n_glen
R0 = hb.R0; H0 = hb.H0; co = parse(Float64, opts["cutoff"])
prof(xi) = xi >= 1 ? 0.0 : H0 * (1 - xi^((n + 1) / n))^(n / (2n + 1))
# margins R(theta) and an interior modulation (a non-separable sheet: dome shape varies with theta)
cases = Dict(
    "circle"    => (th -> R0, (xi, th) -> 1.0),
    "wavy10"    => (th -> R0 * (1 + 0.1cos(3th)), (xi, th) -> 1.0),
    "irregular" => (th -> R0 * (1 + 0.1cos(3th) + 0.05sin(5th) + 0.03cos(7th + 1)), (xi, th) -> 1 + 0.1xi^2 * cos(2th) * (1 - xi)),
)
println("case       bits  grid     chi(H) Landau  chi(D) Landau   chi(H) Cart  chi(D) Cart")
for nb in parse.(Int, split(opts["bits"], ",")), name in ["circle", "wavy10", "irregular"]
    Rf, shape = cases[name]; N = 2^nb
    # Landau grid: xi in (0, 1) cell centred, theta periodic
    hx = 1 / N; ht = 2pi / N
    xi = [(i - 0.5) * hx for i in 1:N]; th = [(j - 1) * ht for j in 1:N]
    H = [prof(x) * shape(x, t) for x in xi, t in th]
    # |grad s|^2 at cell centres (central differences in xi, theta; one-sided at the ends)
    Hxi = similar(H); Hth = similar(H)
    for j in 1:N, i in 1:N
        Hxi[i, j] = (H[min(i + 1, N), j] - H[max(i - 1, 1), j]) / ((min(i + 1, N) - max(i - 1, 1)) * hx)
        Hth[i, j] = (H[i, mod1(j + 1, N)] - H[i, mod1(j - 1, N)]) / (2ht)
    end
    Rv = Rf.(th); Rp = [(Rf(t + 1e-6) - Rf(t - 1e-6)) / 2e-6 for t in th]
    # r = xi R: dH/dr = H_xi / R; (1/r) dH/dtheta|_r = (H_th - xi R'/R H_xi) / (xi R)
    g2 = [(Hxi[i, j] / Rv[j])^2 + ((Hth[i, j] - xi[i] * Rp[j] / Rv[j] * Hxi[i, j]) / (xi[i] * Rv[j]))^2 for i in 1:N, j in 1:N]
    D = G .* H .^ (n + 2) .* g2
    gl = QuanticsGrid(nb, :interleaved, (0.0, 1.0), (0.0, 2pi))
    chH = maxlinkdim(array_to_mps(H, gl; cutoff = co)); chD = maxlinkdim(array_to_mps(D, gl; cutoff = co))
    # same sheet on a Cartesian grid
    Ld = 1.3 * R0; hc = 2Ld / N; xc = [-Ld + (i - 0.5) * hc for i in 1:N]
    Hc = [begin r = hypot(x, y); t = mod(atan(y, x), 2pi); xv = r / Rf(t); prof(xv) * shape(min(xv, 1.0), t) end for x in xc, y in xc]
    Dc = zeros(N, N)
    for j in 2:N-1, i in 2:N-1
        Dc[i, j] = G * Hc[i, j]^(n + 2) * (((Hc[i+1, j] - Hc[i-1, j]) / 2hc)^2 + ((Hc[i, j+1] - Hc[i, j-1]) / 2hc)^2)
    end
    gc = QuanticsGrid(nb, :interleaved, (-Ld, Ld), (-Ld, Ld))
    @printf("%-10s %4d  %4d^2   %8d       %8d        %8d     %8d\n", name, nb, N, chH, chD,
            maxlinkdim(array_to_mps(Hc, gc; cutoff = co)), maxlinkdim(array_to_mps(Dc, gc; cutoff = co)))
    flush(stdout)
end
