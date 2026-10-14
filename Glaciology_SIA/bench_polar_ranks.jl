# Bond dimensions of H, the face diffusivities and the SIA operator on a polar (r, theta) grid vs a
# Cartesian grid of the same number of points, for three ice sheets:
#   radial   Halfar dome at t0 (margin a circle)
#   wavy     same profile with margin R(theta) = R0 (1 + eps cos 3 theta)
#   offset   Halfar dome centred at (x0, 0): the margin is a circle not centred on the pole
#   soft     radial dome with the margin singularity smoothed over a width w_km (softplus of the
#            Halfar base, a thin exponentially decaying film outside): does a margin of fixed physical
#            width give ranks that saturate once the grid resolves it?
#   softwavy the same, wavy margin
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_polar_ranks.jl bits=5,6,7,8
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl"); include("polar.jl")

opts = Dict("bits" => "5,6,7,8", "grids" => "polar,cartesian", "cases" => "radial,wavy,offset", "eps" => "0.1", "x0" => "200e3",
            "tci_tol" => "1e-8", "w_km" => "10", "ops" => "1", "rc" => "1e-12")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
epsw = parse(Float64, opts["eps"]); x0 = parse(Float64, opts["x0"]); tol = parse(Float64, opts["tci_tol"])
p = SIAParams(); hb = Halfar(d = 2, p = p); t0 = halfar_t0(hb); dt = 100.0
Rdom = 1200e3

# thickness as a function of Cartesian position
Hfun = Dict("radial" => (x, y) -> halfar_H(hb, t0, hypot(x, y)),
            "wavy"   => (x, y) -> halfar_H(hb, t0, hypot(x, y) / (1 + epsw * cos(3 * atan(y, x)))),
            "offset" => (x, y) -> halfar_H(hb, t0, hypot(x - x0, y)))
# softplus-smoothed Halfar profile: margin width ~ (3/4) w R0 in r
wsoft = parse(Float64, opts["w_km"]) * 1e3 / (0.75 * hb.R0)
function soft_H(r)
    n = p.n_glen; tau = t0 / t0
    inner = 1 - (r / hb.R0)^((n + 1) / n)
    sp = inner / wsoft > 40 ? inner : wsoft * log1p(exp(inner / wsoft))
    return hb.H0 * sp^(n / (2n + 1))
end
Hfun["soft"] = (x, y) -> soft_H(hypot(x, y))
Hfun["softwavy"] = (x, y) -> soft_H(hypot(x, y) / (1 + epsw * cos(3 * atan(y, x))))

rc = parse(Float64, opts["rc"]); doops = opts["ops"] == "1"
# chi(H~), chi(D~): after SVD recompression at relative cutoff rc (removes TCI / arithmetic noise)
println("grid       case    bits  points      chi(H)  chi(D1) chi(D2) chi(L)  chi(A)   H numbers  x compr   t [s]  chi(H~) chi(D1~) chi(D2~)")
for nb in bits, grid in split(opts["grids"], ","), case in split(opts["cases"], ",")
    st = SolverSettings(cutoff = 1e-13, maxdim = 1024, op_maxdim = 4096)
    t = @elapsed begin
        if grid == "polar"
            g = QuanticsGrid(nb, :interleaved, (0.0, Rdom), (0.0, 2pi))
            po = PolarOperators(g)
            H = function_to_mps((r, th) -> Hfun[case](r * cos(th), r * sin(th)), g; tolerance = tol, maxbonddim = 512)
            D = polar_face_diffusivities(H, po, p, st)
            L = polar_diffusion_mpo(D, po, st)
            Iop = po.I
        else
            g = QuanticsGrid(nb, :interleaved, (-Rdom, Rdom), (-Rdom, Rdom))
            ops = SIAOperators(g)
            H = function_to_mps((x, y) -> Hfun[case](x, y), g; tolerance = tol, maxbonddim = 512)
            D = face_diffusivities(H, nothing, ops, p, st)
            L = doops ? diffusion_mpo(D, ops, st) : nothing
            Iop = ops.I
        end
        A = L === nothing ? nothing : mpo_lincomb([1.0, -dt], [Iop, L]; cutoff = st.op_cutoff)
    end
    tr(x) = maxlinkdim(truncate(x; cutoff = rc))
    println(@sprintf("%-10s %-7s %4d  %4d^2  %8d  %7d %7d %7d %7d  %10d  %8.3g  %6.1f  %7d %8d %8d", grid, case, nb, 2^nb, maxlinkdim(H), maxlinkdim(D[1]),
                     maxlinkdim(D[2]), L === nothing ? 0 : maxlinkdim(L), A === nothing ? 0 : maxlinkdim(A), nparams(H), compression(H, g), t,
                     tr(H), tr(D[1]), tr(D[2])))
    flush(stdout)
end
