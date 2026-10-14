# Figures for the note (run with the Shakti environment, which has CairoMakie; read-only use):
#   julia --project=/albedo/home/taange001/Shakti.jl plot_figures.jl
# Writes figures/hydro_*.png here and copies them to ~/PostDoc_Latex_Notes/MPS/figures/.

using CairoMakie, HDF5, Printf

const DIR = @__DIR__
const FIG = joinpath(DIR, "figures"); mkpath(FIG)
const NOTEFIG = expanduser("~/PostDoc_Latex_Notes/MPS/figures")

function load(file, day)
    key = @sprintf("day_%05.1f", day)
    h5open(file, "r") do f
        g = f[key]
        qx = read(g["q_x"]); qy = read(g["q_y"]); n = size(qx, 2)
        q = sqrt.(((qx[1:n, :] .+ qx[2:n+1, :]) ./ 2) .^ 2 .+ ((qy[:, 1:n] .+ qy[:, 2:n+1]) ./ 2) .^ 2)
        (x = read(f["x"]), y = read(f["y"]), b = read(g["b"]), N = read(g["N"]), h = read(g["h"]), q = q, K = read(g["K"]))
    end
end

# --- 1. fields -----------------------------------------------------------------------------------
function fig_fields(file, days, out)
    fig = Figure(size = (1500, 380 * length(days)))
    for (r, d) in enumerate(days)
        s = load(file, d)
        x = s.x ./ 1000; y = s.y ./ 1000
        panels = [("log10 gap height b [m]", log10.(max.(s.b, 1e-3)), :inferno),
                  ("effective pressure N [MPa]", s.N ./ 1e6, :viridis),
                  ("hydraulic head h [m]", s.h, :viridis),
                  ("log10 |q| [m²/s]", log10.(max.(s.q, 1e-8)), :plasma)]
        for (c, (t, A, cm)) in enumerate(panels)
            ax = Axis(fig[r, 2c-1], title = "day $(round(Int, d)): $t", xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
            hm = heatmap!(ax, x[2:end-1], y[2:end-1], A[2:end-1, 2:end-1], colormap = cm)   # rim cells are solver bookkeeping
            Colorbar(fig[r, 2c], hm)
        end
    end
    save(out, fig); println("wrote $out")
end

# --- 2. gap height at several resolutions --------------------------------------------------------
function fig_resolutions(files, day, out)
    fig = Figure(size = (400 * length(files), 520))
    for (c, f) in enumerate(files)
        s = load(f, day)
        ax = Axis(fig[1, c], title = "N = $(length(s.x)), day $(round(Int, day))", xlabel = "x [km]", ylabel = "y [km]", aspect = DataAspect())
        hm = heatmap!(ax, s.x ./ 1000, s.y ./ 1000, log10.(max.(s.b, 1e-3)), colormap = :inferno, colorrange = (-3, -0.8))
        c == length(files) && Colorbar(fig[1, c+1], hm, label = "log10 b [m]")
    end
    save(out, fig); println("wrote $out")
end

# --- results parsing -----------------------------------------------------------------------------
function rows(files, kind)
    out = Vector{Vector{String}}()
    for f in files
        isfile(f) || continue
        for l in eachline(f)
            p = split(l)
            !isempty(p) && p[1] == kind && push!(out, String.(p))
        end
    end
    out
end

# --- 3. rank vs resolution -----------------------------------------------------------------------
function fig_ranks(resfiles, out; eps = "0.001", ord = "interleaved")
    R = rows(resfiles, "RANK")
    fig = Figure(size = (1100, 450))
    ax1 = Axis(fig[1, 1], title = "Shakti seasonal run, channelized (day 200), eps = $eps, $ord", xlabel = "n  (grid 2^n x 2^n)",
               ylabel = "max QTT rank", yscale = log2)
    for (name, col) in (("b", :red), ("logb", :orange), ("qmag", :purple), ("K", :brown), ("N", :blue), ("h", :green))
        for (lab, ls) in (("day200", :solid), ("day120", :dash))
            pts = sort([(parse(Int, r[4]), parse(Int, r[7])) for r in R if startswith(r[2], "N") && occursin(lab, r[2]) &&
                        !occursin("diff", r[2]) && r[3] == name && r[5] == ord && r[6] == eps])
            isempty(pts) && continue
            lines!(ax1, first.(pts), last.(pts), color = col, linestyle = ls, label = lab == "day200" ? name : nothing)
            scatter!(ax1, first.(pts), last.(pts), color = col)
        end
    end
    nn = 6:12; lines!(ax1, nn, 2.0 .^ (nn ./ 2), color = :gray, linestyle = :dot, label = "2^(n/2)")
    lines!(ax1, nn, 2.0 .^ nn, color = :black, linestyle = :dot, label = "2^n (full)")
    axislegend(ax1, position = :lt)
    ax2 = Axis(fig[1, 2], title = "synthetic channel networks, eps = $eps, $ord", xlabel = "n", ylabel = "max QTT rank", yscale = log2)
    for (lab, col) in (("syn_dendritic_phys25", :blue), ("syn_dendritic_phys8", :purple), ("syn_dendritic_grid", :red), ("syn_straight_grid", :green))
        pts = sort([(parse(Int, r[4]), parse(Int, r[7])) for r in R if r[2] == lab && r[5] == ord && r[6] == eps])
        isempty(pts) && continue
        lines!(ax2, first.(pts), last.(pts), color = col, label = replace(lab, "syn_" => ""))
        scatter!(ax2, first.(pts), last.(pts), color = col)
    end
    lines!(ax2, nn, 2.0 .^ (nn ./ 2), color = :gray, linestyle = :dot, label = "2^(n/2)")
    axislegend(ax2, position = :lt)
    save(out, fig); println("wrote $out")
end

# --- 4. QTT vs AMR: degrees of freedom vs max error ----------------------------------------------
function fig_qtt_amr(resfiles, cases, out)
    Q = rows(resfiles, "QTT"); A = rows(resfiles, "AMR")
    fig = Figure(size = (450 * length(cases), 420))
    for (c, (lab, name, n)) in enumerate(cases)
        ax = Axis(fig[1, c], title = "$lab $name (n = $n)", xlabel = "relative max error", ylabel = "DOFs", xscale = log10, yscale = log10)
        for (ord, col) in (("interleaved", :red), ("sequential", :orange))
            pts = [(parse(Float64, r[10]), parse(Float64, r[8])) for r in Q if r[2] == lab && r[3] == name && r[4] == string(n) && r[5] == ord]
            isempty(pts) && continue
            pts = sort(pts); scatterlines!(ax, first.(pts), last.(pts), color = col, label = "QTT $ord (params)")
        end
        pts = sort([(parse(Float64, r[10]), parse(Float64, r[6])) for r in A if r[2] == lab && r[3] == name && r[4] == string(n)])
        isempty(pts) || scatterlines!(ax, first.(pts), last.(pts), color = :blue, label = "AMR quadtree (cells)")
        hlines!(ax, [4.0^n], color = :black, linestyle = :dot, label = "uniform grid")
        c == 1 && axislegend(ax, position = :lb)
    end
    save(out, fig); println("wrote $out")
end

# --- 5. bond-dimension profiles ------------------------------------------------------------------
function fig_profiles(resfiles, cases, out; eps = "0.0001")
    R = rows(resfiles, "RANK")
    fig = Figure(size = (900, 400))
    ax = Axis(fig[1, 1], title = "bond dimensions, eps = $eps", xlabel = "bond k", ylabel = "rank", yscale = log2)
    for (lab, name, n, col) in cases, (ord, ls) in (("interleaved", :solid), ("sequential", :dash))
        r = [x for x in R if x[2] == lab && x[3] == name && x[4] == string(n) && x[5] == ord && x[6] == eps]
        isempty(r) && continue
        prof = parse.(Int, split(r[1][8], ","))
        lines!(ax, 1:length(prof), prof, color = col, linestyle = ls, label = "$lab $name $ord")
    end
    axislegend(ax, position = :rt, labelsize = 9)
    save(out, fig); println("wrote $out")
end

data(N; tag = "") = joinpath(DIR, "data", "seasonal_N$(N)$(tag).h5")
resfiles = filter(isfile, [joinpath(DIR, "results", f) for f in readdir(joinpath(DIR, "results"))])
what = isempty(ARGS) ? ["all"] : ARGS

for N in (256, 512)
    if isfile(data(N)) && ("all" in what || "fields" in what)
        days = [d for d in (120.0, 185.0, 200.0) if h5open(f -> haskey(f, @sprintf("day_%05.1f", d)), data(N), "r")]
        fig_fields(data(N), days, joinpath(FIG, "hydro_fields_N$N.png"))
    end
end
if "all" in what || "res" in what
    fs = filter(isfile, data.((64, 128, 256, 512)))
    fig_resolutions(fs, 200.0, joinpath(FIG, "hydro_b_resolutions.png"))
end
if "all" in what || "ranks" in what
    fig_ranks(resfiles, joinpath(FIG, "hydro_ranks_vs_n.png"))
    fig_ranks(resfiles, joinpath(FIG, "hydro_ranks_vs_n_1e-2.png"); eps = "0.01")
    fig_qtt_amr(resfiles, [("N256_day200", "b", 8), ("N256_day200", "qmag", 8), ("syn_dendritic_phys25", "b", 11), ("syn_dendritic_grid", "b", 11)],
                joinpath(FIG, "hydro_qtt_vs_amr.png"))
    fig_profiles(resfiles, [("N256_day200", "b", 8, :red), ("N256_day200", "h", 8, :green), ("N256_day120", "b", 8, :blue)],
                 joinpath(FIG, "hydro_bond_profiles.png"))
end
isdir(NOTEFIG) && for f in readdir(FIG)
    cp(joinpath(FIG, f), joinpath(NOTEFIG, f); force = true)
end
