# Figures for the LaTeX note, from the closed form and the RESULT lines of the job logs.
#   sbatch slurm/run.sbatch plots.jl f1b=slurm/logs/f1b_XXXX.out step=slurm/logs/...,... out=/albedo/home/taange001/PostDoc_Latex_Notes/MPS/figures
using Plots, Printf
gr()
include("fabric_common.jl")

opts = parse_opts(Dict("f1b" => "", "step" => "", "sep" => "", "out" => "figures", "wb" => "0.35", "X" => "4.0"))
out = opts["out"]; mkpath(out)
kv(line, key) = (m = match(Regex("\\b$key=([^ ]+)"), line); m === nothing ? nothing : m.captures[1])

# ---- 1. example fabrics psi(phi, mu) and the a1 field of the divide flow
let
    f = DivideFlow(wb = parse(Float64, opts["wb"])); X = parse(Float64, opts["X"])
    Nx = Nz = 64; xs = cellc(0.0, X, Nx); zs = cellc(0.0, 1.0, Nz)
    Ms = [Mfrom(path_G(f, x, z)) for x in xs, z in zs]
    xq, wq = gauss_legendre(64); pq = phi_grid(128)
    a1 = [a2_eigs([psi_closed(Ms[i, j], m, p) for m in xq, p in pq], xq, wq, pq)[1][1] for i in 1:Nx, j in 1:Nz]
    p1 = heatmap(xs, zs, a1', xlabel = "x / H", ylabel = "z / H", title = "largest eigenvalue a1 of <n n>", c = :viridis, colorbar = true)
    mus = mu_grid(128); phs = phi_grid(256)
    pts = [(8, 56), (32, 16), (60, 3)]
    ps = [p1]
    for (i, j) in pts
        A = [psi_closed(Ms[i, j], m, p) for m in mus, p in phs]
        push!(ps, heatmap(rad2deg.(phs), mus, A, xlabel = "phi (deg, from +x towards +z)", ylabel = "mu = n_y", c = :inferno,
                          title = @sprintf("psi at x=%.2f, z=%.2f (a1=%.2f)", xs[i], zs[j], a1[i, j]), titlefontsize = 9))
        scatter!(p1, [xs[i]], [zs[j]], label = "", mc = :red, ms = 4)
    end
    plot(ps..., layout = (2, 2), size = (1100, 800), left_margin = 3Plots.mm, bottom_margin = 3Plots.mm)
    savefig(joinpath(out, "fabric_field_examples.png"))
    println("wrote fabric_field_examples.png")
end

# ---- 2. storage vs fabric sharpness (test 1b)
if opts["f1b"] != ""
    lines = readlines(opts["f1b"])
    hw = Dict{String,Float64}(); a1 = Dict{String,Float64}()
    for l in lines
        m = match(r"^=== (\w+) s=([-0-9.]+): a = \(([0-9.]+),.*half-width ([0-9.]+) deg", l)
        m === nothing && continue
        key = m.captures[1] * ":" * string(parse(Float64, m.captures[2]))
        hw[key] = parse(Float64, m.captures[4]); a1[key] = parse(Float64, m.captures[3])
    end
    for e in ("0.0001", "0.01")
        rows = [l for l in lines if startswith(l, "RESULT_1B") && occursin("n=10 ", l) && occursin("eps=$e ", l)]
        data = map(rows) do l
            w = split(l); key = w[2] * ":" * string(parse(Float64, w[3]))
            (key = key, kind = w[2], hw = hw[key], sh = parse(Int, kv(l, "sh_par")), qs = parse(Int, kv(l, "qtts_par")),
             qi = parse(Int, kv(l, "qtti_par")), r = parse(Int, kv(l, "svd_rank")))
        end
        acg = filter(d -> d.kind != "watson", data); wat = filter(d -> d.kind == "watson" && !occursin("-", d.key), data)
        sort!(acg, by = d -> -d.hw); sort!(wat, by = d -> -d.hw)
        p = plot(xscale = :log10, yscale = :log10, xflip = true, xlabel = "ODF half-width (deg)", ylabel = "numbers stored per fabric",
                 title = "one fabric, rel. L2 error $(e == "0.0001" ? "1e-4" : "1e-2")", legend = :topleft, size = (700, 500))
        scatter!(p, [d.hw for d in acg], [d.sh for d in acg], label = "SH, lattice rotation (ACG)", m = :circle, mc = 1)
        scatter!(p, [d.hw for d in acg], [d.qs for d in acg], label = "QTT (mu bits | phi bits), ACG", m = :diamond, mc = 2)
        scatter!(p, [d.hw for d in wat], [d.sh for d in wat], label = "SH, Watson", m = :circle, mc = 1, ma = 0.4)
        scatter!(p, [d.hw for d in wat], [d.qs for d in wat], label = "QTT, Watson", m = :diamond, mc = 2, ma = 0.4)
        savefig(p, joinpath(out, "fabric_storage_vs_width_eps$(e == "0.0001" ? "1e-4" : "1e-2").png"))
        println("wrote storage plot eps=$e")
    end
end

# ---- 3. space|orientation singular values (test 1)
if opts["sep"] != ""
    p = plot(yscale = :log10, xlabel = "k", ylabel = "sigma_k / sigma_1", title = "space | orientation SVD of psi(x,z,mu,phi)", size = (700, 450))
    for fn in split(opts["sep"], ",")
        for l in readlines(fn)
            startswith(l, "RESULT_SEPSPEC") || continue
            w = split(l); vals = parse.(Float64, split(w[end], ","))
            plot!(p, 1:length(vals), max.(vals, 1e-17), label = join(w[2:4], " "), lw = 2)
        end
    end
    savefig(p, joinpath(out, "fabric_separation_spectrum.png"))
    println("wrote separation spectrum")
end

# ---- 4. stepper cost vs size
if opts["step"] != ""
    mps = Dict{Int,Float64}(); den = Dict{Int,Float64}()
    for fn in split(opts["step"], ","), l in readlines(fn)
        if startswith(l, "RESULT_STEP")
            mps[parse(Int, kv(l, "Ntot"))] = parse(Float64, kv(l, "mps_s_per_step"))
        elseif startswith(l, "RESULT_DENSE")
            den[parse(Int, kv(l, "Ntot"))] = parse(Float64, kv(l, "dense_s_per_step"))
        end
    end
    p = plot(xscale = :log2, yscale = :log10, xlabel = "degrees of freedom", ylabel = "seconds per RK4 step", legend = :topleft, size = (700, 450))
    k = sort(collect(keys(mps))); scatter!(p, k, [mps[i] for i in k], label = "MPS (MPO x MPS)", m = :diamond)
    k = sort(collect(keys(den))); scatter!(p, k, [den[i] for i in k], label = "dense", m = :circle)
    savefig(p, joinpath(out, "fabric_stepper_cost.png"))
    println("wrote stepper cost")
end
println("done")
