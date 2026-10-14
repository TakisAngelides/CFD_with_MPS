# Resolution scan: summary table and plots from several run_sia.jl outputs (same experiment)
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch plot_scan.jl out.png output/a.h5 output/b.h5 ...
ENV["GKSwstype"] = "100"
using HDF5, Plots, Printf, Statistics
nparams_from_links(links, sitedim = 2) = sum(sitedim * l * r for (l, r) in zip([1; links], [links; 1]))

png = ARGS[1]
rows = []
for fname in ARGS[2:end]
    h5open(fname, "r") do f
        a = Dict(k => read_attribute(f, k) for k in keys(attributes(f)))
        times = read(f, "times"); wall = read(f, "walltime")
        dts = haskey(f, "dt") ? read(f, "dt") : fill(a["dt"], length(wall))
        dg = Dict(k => read(f, "diag/$(k)") for k in keys(f["diag"]))
        L = a["experiment"] == "halfar" ? 1200e3 : 750e3
        push!(rows, (; nbits = a["nbits"], h = 2L / 2^a["nbits"], t = times[end] - times[1], steps = length(wall),
                     wall_total = sum(wall), wall_step = mean(wall), dt_mean = mean(dts), dt_last = dts[end],
                     chiH = maximum(read(f, "chi_H")), chiA = maximum(read(f, "chi_A")),
                     ngrid = 2.0^(a["d"] * a["nbits"]),
                     parH = haskey(f, "params_H") ? read(f, "params_H")[end] : nparams_from_links(read(f, "linkdims_final")),
                     parA = haskey(f, "params_A") ? maximum(read(f, "params_A")) : NaN,
                     stencil = haskey(attributes(f), "op_stencil") ? read_attribute(f, "op_stencil") : (a["d"] == 1 ? 3 : 9),
                     err_max = haskey(dg, "err_max") ? dg["err_max"][end] : NaN,
                     err_ctr = haskey(dg, "err_center") ? dg["err_center"][end] : NaN,
                     dV = haskey(dg, "err_volume_rel") ? dg["err_volume_rel"][end] : NaN,
                     Hc = dg["H_center"][end],
                     margin = haskey(f, "margin_model") ? read(f, "margin_model") : NaN,
                     margin_exact = haskey(f, "margin_exact") ? read(f, "margin_exact") : NaN,
                     finished = haskey(attributes(f), "finished") ? read_attribute(f, "finished") : true))
    end
end
sort!(rows, by = r -> r.nbits)
println("| points | h (m) | simulated (a) | steps | mean dt (a) | s/step | total (min) | chi(H) | H numbers | H compression | chi(A) | operator compression vs sparse | max err (m) | dome err (m) | dV/V | dome (m) | margin (km) |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows
    println(@sprintf("| 2^%d | %.1f | %.0f | %d | %.3g | %.1f | %.1f | %d | %d | x%.3g | %d | %s | %.3g | %.3g | %.1e | %.1f | %.2f |%s",
                     r.nbits, r.h, r.t, r.steps, r.dt_mean, r.wall_step, r.wall_total / 60, r.chiH, r.parH, r.ngrid / r.parH, r.chiA,
                     isnan(r.parA) ? "-" : @sprintf("x%.3g", r.stencil * r.ngrid / r.parA), r.err_max, r.err_ctr, r.dV, r.Hc,
                     r.margin / 1e3, r.finished ? "" : " (unfinished)"))
end
nb = [r.nbits for r in rows]
p1 = if all(isnan, [r.err_max for r in rows])
    p = plot(nb, [r.margin / 1e3 for r in rows], marker = :o, xlabel = "bits", ylabel = "margin (km)", label = "model")
    hline!(p, [rows[1].margin_exact / 1e3], ls = :dash, label = "mass-balance closure")
    p
else
    p = plot([r.h for r in rows], [r.err_max for r in rows], marker = :o, xscale = :log10, yscale = :log10, xlabel = "h (m)", ylabel = "error (m)", label = "max error")
    plot!(p, [r.h for r in rows], [r.err_ctr for r in rows] .+ 1e-4, marker = :s, label = "dome error")
    p
end
p2 = plot(nb, [r.wall_step for r in rows], marker = :o, yscale = :log10, xlabel = "bits", ylabel = "s / step", label = "wall time per step")
p3 = plot(nb, [r.chiH for r in rows], marker = :o, xlabel = "bits", ylabel = "max bond dimension", label = "chi(H)")
plot!(p3, nb, [r.chiA for r in rows], marker = :s, label = "chi(operator)")
p4 = plot(nb, [r.ngrid / r.parH for r in rows], marker = :o, yscale = :log10, xlabel = "bits", ylabel = "compression factor", label = "H: grid values / MPS numbers", legend = :topleft)
plot!(p4, nb, [r.stencil * r.ngrid / r.parA for r in rows], marker = :s, label = "operator: sparse nnz / MPO numbers")
savefig(plot(p1, p2, p3, p4, layout = (2, 2), size = (1100, 800)), png)
println("wrote ", png)
