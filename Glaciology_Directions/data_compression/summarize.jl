# summarize.jl -- figures (figures/compress_*.png) and LaTeX table rows (results/tables.tex)
# from the CSV files written by run_compress.jl
ENV["GKSwstype"] = "100"
using Plots, Printf, Statistics, HDF5
include("qttlib.jl")

const RES = joinpath(@__DIR__, "results")
const FIG = joinpath(@__DIR__, "figures")
mkpath(FIG)
default(fontfamily = "Computer Modern", framestyle = :box, grid = true, legendfontsize = 7, guidefontsize = 9, tickfontsize = 7, titlefontsize = 9, dpi = 200)

function readcsv(file)
    lines = readlines(file)
    hdr = split(lines[1], ",")
    rows = [Dict(zip(hdr, split(l, ","))) for l in lines[2:end] if !startswith(l, "#") && !isempty(l)]
    return rows
end
num(s) = isempty(s) ? NaN : parse(Float64, s)

const STYLE = Dict("qtt_int" => (:black, :circle, "QTT interleaved"), "qtt_seq" => (:gray, :utriangle, "QTT sequential"),
                   "svd" => (:red, :square, "matrix SVD"), "haar" => (:blue, :diamond, "Haar wavelet"),
                   "cdf97" => (:green, :star5, "CDF 9/7 wavelet"), "zlib" => (:orange, :xcross, "quantised + zlib"))
const METHODS = ["qtt_int", "qtt_seq", "svd", "haar", "cdf97", "zlib"]

# log-log interpolation of ratio (storage / cells) at a target RMS error
function ratio_at(rows, method, target; key = "rms")
    pts = sort([(num(r[key]), num(r["bytes"]) / 4 / (2.0^(2 * parse(Int, r["n"])))) for r in rows if r["method"] == method && num(r[key]) > 0]; by = first)
    isempty(pts) && return NaN
    x = log10.(first.(pts)); y = log10.(last.(pts)); t = log10(target)
    (t < x[1] || t > x[end]) && return NaN
    k = searchsortedlast(x, t)
    k == length(x) && return 10^y[end]
    w = (t - x[k]) / (x[k+1] - x[k])
    return 10^((1 - w) * y[k] + w * y[k+1])
end

const FIELDS = [("grl_bed", "Greenland bed (150 m)", "m"), ("grl_thk", "Greenland thickness (raw)", "m"),
                ("grl_thk_ice", "Greenland thickness (ice, filled)", "m"), ("grl_srf", "Greenland surface (raw)", "m"),
                ("grl_srf_ice", "Greenland surface (ice, filled)", "m"), ("grl_mask", "Greenland ice mask", "-"),
                ("ant_speed", "Antarctic speed (450 m)", "m/a"), ("ant_logspeed", "Antarctic log10 speed", "log10(m/a)"),
                ("ant_bed", "Antarctic bed (2 km)", "m")]
const TARGETS = Dict("grl_bed" => [3.0, 10.0, 30.0], "grl_thk" => [3.0, 10.0, 30.0], "grl_thk_ice" => [3.0, 10.0, 30.0],
                     "grl_srf" => [1.0, 3.0, 10.0], "grl_srf_ice" => [1.0, 3.0, 10.0], "grl_mask" => [0.03, 0.1],
                     "ant_speed" => [1.0, 3.0, 10.0], "ant_logspeed" => [0.01, 0.03, 0.1], "ant_bed" => [10.0, 30.0, 100.0])

tag = get(Dict(String(split(a, "=")[1]) => String(split(a, "=")[2]) for a in ARGS), "tag", "n13")

# ------------------------------------------------------------------ rate-distortion figures + table
tab = open(joinpath(RES, "tables_$(tag).tex"), "w")
plots_rd = []
for (field, title, unit) in FIELDS
    t = field == "ant_bed" ? "n11" : tag
    file = joinpath(RES, "rd_$(field)_$(t).csv")
    isfile(file) || continue
    rows = readcsv(file)
    p = plot(xscale = :log10, yscale = :log10, xlabel = "RMS error [$unit]", ylabel = "storage / cells", title = title, legend = :bottomleft)
    for m in METHODS
        pts = sort([(num(r["rms"]), num(r["bytes"]) / 4 / 2.0^(2 * parse(Int, r["n"]))) for r in rows if r["method"] == m && num(r["rms"]) > 0]; by = first)
        isempty(pts) && continue
        c, mk, lab = STYLE[m]
        plot!(p, first.(pts), last.(pts), color = c, marker = mk, ms = 3, label = lab)
    end
    hline!(p, [1.0], color = :black, ls = :dash, label = "")
    push!(plots_rd, p)
    savefig(p, joinpath(FIG, "compress_rd_$(field).png"))
    for target in TARGETS[field]
        vals = [ratio_at(rows, m, target) for m in METHODS]
        str(v) = isnan(v) ? "--" : v >= 1 ? @sprintf("%.2f", v) : v >= 0.01 ? @sprintf("%.3f", v) : @sprintf("%.1e", v)
        println(tab, replace(title, "_" => "\\_"), " & ", target, " & ", join(str.(vals), " & "), " \\\\")
    end
end
close(tab)
length(plots_rd) > 0 && savefig(plot(plots_rd..., layout = (cld(length(plots_rd), 3), 3), size = (1200, 380 * cld(length(plots_rd), 3))), joinpath(FIG, "compress_rd_all.png"))

# ------------------------------------------------------------------ rank vs cutoff / bond profiles
for (field, title, unit) in FIELDS
    file = joinpath(RES, "rd_$(field)_$(field == "ant_bed" ? "n11" : tag).csv")
    isfile(file) || continue
    rows = [r for r in readcsv(file) if startswith(r["method"], "qtt") && !isempty(r["ranks"])]
    isempty(rows) && continue
    p = plot(yscale = :log10, xlabel = "bond", ylabel = "bond dimension", title = title, legend = :topright)
    for r in rows
        rk = parse.(Int, split(r["ranks"], ";"))
        ls = r["method"] == "qtt_int" ? :solid : :dash
        plot!(p, rk, ls = ls, label = @sprintf("%s eps=%s", r["method"][5:7], r["param"]), lw = 1)
    end
    savefig(p, joinpath(FIG, "compress_bonds_$(field).png"))
end

# ------------------------------------------------------------------ scaling with resolution
for (field, title, unit) in FIELDS
    files = filter(f -> startswith(f, "scale_$(field)_n"), readdir(RES))
    isempty(files) && continue
    rows = readcsv(joinpath(RES, sort(files)[end]))
    ps = []
    for kind in ("coarse", "crop")
        p = plot(yscale = :log10, xlabel = kind == "coarse" ? "log2 N (fixed extent, coarsened)" : "log2 N (native resolution, growing crop)",
                 ylabel = "max bond dim / SVD rank", title = title, legend = :topleft)
        for (mi, m) in enumerate(["qtt_int", "qtt_seq", "svd"])
            for (ti, target) in enumerate(TARGETS[field])
                pts = sort([(parse(Int, r["m"]), num(r["maxrank"])) for r in rows if r["kind"] == kind && r["method"] == m && num(r["target"]) ≈ target && num(r["maxrank"]) > 0])
                isempty(pts) && continue
                c, mk, lab = STYLE[m]
                plot!(p, first.(pts), last.(pts), color = c, marker = mk, ms = 3, ls = [:solid, :dash, :dot][ti], label = "$(lab), rms $(target)")
            end
        end
        mm = 8:13
        plot!(p, mm, 2.0 .^ (mm ./ 2), color = :black, ls = :dashdot, lw = 0.6, label = "2^(n/2)")
        push!(ps, p)
    end
    savefig(plot(ps..., layout = (1, 2), size = (1100, 420)), joinpath(FIG, "compress_scale_$(field).png"))
end

# ------------------------------------------------------------------ overview maps
function blockmean(A, f)
    N = size(A, 1) ÷ f
    [mean(@view A[(i-1)*f+1:i*f, (j-1)*f+1:j*f]) for i in 1:N, j in 1:N]
end
if isfile(joinpath(@__DIR__, "data", "grl8192.h5"))
    h = h5open(joinpath(@__DIR__, "data", "grl8192.h5"), "r")
    ms = []
    for (k, lab) in (("bed", "bed [m]"), ("thickness", "thickness [m]"), ("surface", "surface [m]"))
        A = blockmean(Float64.(read(h[k])), 16)
        push!(ms, heatmap(A', yflip = true, aspect_ratio = 1, title = "Greenland $(lab), 150 m crop", c = :viridis, axis = false))
    end
    close(h)
    s = Float64.(h5read(joinpath(@__DIR__, "data", "antvel8192.h5"), "speed"))
    ls = log10.(max.(ifelse.(isfinite.(s), s, NaN), 0.1))
    L = [let v = filter(isfinite, vec(ls[(i-1)*16+1:i*16, (j-1)*16+1:j*16])); isempty(v) ? NaN : mean(v) end for i in 1:512, j in 1:512]
    push!(ms, heatmap(L', yflip = true, aspect_ratio = 1, title = "Antarctic log10 speed [m/a], 450 m crop", c = :magma, axis = false))
    savefig(plot(ms..., layout = (2, 2), size = (1000, 900)), joinpath(FIG, "compress_maps.png"))
end
println("summary done")
