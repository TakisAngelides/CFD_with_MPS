# Plot an output file of run_sia.jl (headless, writes PNGs next to the .h5 file)
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch plot_sia.jl output/halfar_d2_....h5

ENV["GKSwstype"] = "100"
using HDF5, Plots
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

for fname in ARGS
    f = h5open(fname, "r")
    attrs = Dict(k => read_attribute(f, k) for k in keys(attributes(f)))
    d = attrs["d"]
    times = read(f, "times"); snap_times = read(f, "snap_times")
    snaps = read(f, "snapshots"); x = read(f, "coords/x1") ./ 1e3
    chiH = read(f, "chi_H"); chiA = read(f, "chi_A"); wall = read(f, "walltime")
    diag = Dict(k => read(f, "diag/$(k)") for k in keys(f["diag"]))
    speed = read(f, "speed_final")
    dts = haskey(f, "dt") ? read(f, "dt") : fill(attrs["dt"], length(times) - 1)
    close(f)
    base = splitext(fname)[1]
    title = "$(attrs["experiment"]) d=$(d) 2^$(attrs["nbits"]) $(attrs["ordering"]) $(attrs["scheme"]) dt=$(attrs["dt"])a"

    # Profiles through the centre at the snapshot times (+ exact solution for Halfar)
    mid = length(x) ÷ 2 + 1
    prof(k) = d == 1 ? snaps[:, k] : snaps[:, mid, k]
    p1 = plot(xlabel = "x (km)", ylabel = "H (m)", title = title, titlefontsize = 8, legend = :topright)
    ks = unique(round.(Int, range(1, length(snap_times), length = min(6, length(snap_times)))))
    for k in ks
        plot!(p1, x, prof(k), label = "t = $(round(Int, snap_times[k])) a")
        if attrs["experiment"] == "halfar"
            hb = Halfar(d = d)
            plot!(p1, x, [halfar_H(hb, snap_times[k], abs(xi * 1e3)) for xi in x], ls = :dash, color = :black, label = "")
        end
    end

    # Diagnostics in time
    p2 = plot(times[2:end], chiH, label = "chi(H)", xlabel = "t (a)", ylabel = "bond dimension")
    plot!(p2, times[2:end], chiA, label = "chi(I - dt L)")
    p3 = plot(times[2:end], wall, xlabel = "t (a)", ylabel = "wall time / step (s)", label = "wall time", legend = :topleft)
    plot!(twinx(p3), times[2:end], dts, color = :red, ylabel = "dt (a)", label = "dt (adaptive)", yscale = :log10, legend = :topright)
    if haskey(diag, "err_max")
        p4 = plot(times, diag["err_max"], label = "max |H - H_exact|", xlabel = "t (a)", ylabel = "m", yscale = :log10)
        plot!(p4, times, diag["err_center"] .+ 1e-3, label = "dome error")
        plot!(p4, times, diag["err_mean_ice"], label = "mean error (ice)")
    else
        p4 = plot(times, diag["H_center"], label = "dome thickness", xlabel = "t (a)", ylabel = "m")
    end
    savefig(plot(p1, p2, p3, p4, layout = (2, 2), size = (1300, 800), left_margin = 5Plots.mm, right_margin = 12Plots.mm), base * "_summary.png")

    if d == 2
        hm1 = heatmap(x, x, snaps[:, :, end]', aspect_ratio = 1, title = "H (m), t = $(round(Int, snap_times[end])) a", xlabel = "x (km)", ylabel = "y (km)")
        hm2 = heatmap(x, x, log10.(speed' .+ 1e-3), aspect_ratio = 1, title = "log10 |u| (m/a)", xlabel = "x (km)")
        savefig(plot(hm1, hm2, layout = (1, 2), size = (1200, 520)), base * "_maps.png")
    end
    println("plotted ", base)
end
