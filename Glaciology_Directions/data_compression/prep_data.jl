# prep_data.jl -- read the native-resolution datasets (read only) and write 2^n x 2^n crops into
# data/ of this folder (Float32 HDF5). Also prints a short survey (sizes, ranges, error levels).
#
#   BedMachine Greenland v5.5 (150 m, 10218 x 18346): bed, thickness, surface, mask
#   MEaSUREs InSAR-based Antarctic ice velocity v2 (450 m, 12445 x 12445): VX, VY -> speed
#   BedMachine Antarctica v3 "sparse" (every 4th point, 2 km, 3334 x 3334): bed
#
# netCDF4 files are read with HDF5.jl; Julia sees the (y, x) netCDF variables as (x, y).

using HDF5, Statistics, Printf

const OUT = joinpath(@__DIR__, "data")
mkpath(OUT)

const GRL = "/albedo/work/user/daabel001/issm/DJ/data/BedMachineGreenland-v5.nc"
const VEL = "/albedo/work/user/daabel001/issm/hipeval/data/antarctica_ice_velocity_450m_v2.nc"
const ANTBED = "/albedo/work/projects/p_forclima/isostasy_data/topography/BedMachineAntarctica-v3-sparse.nc"

function survey(file)
    println("== ", file)
    h5open(file, "r") do h
        for k in keys(h)
            d = h[k]
            d isa HDF5.Dataset || continue
            try
                println("   ", rpad(k, 12), size(d), "  ", eltype(d))
            catch
                println("   ", rpad(k, 12), "(unreadable type)")
            end
        end
    end
end

pct(v, p) = isempty(v) ? NaN : quantile(v, p)

# best 2^n crop offset (0-based) maximising the number of `good` cells, searched on a stride grid
function best_offset(good::AbstractMatrix{Bool}, N::Int; stride::Int = 16)
    S = cumsum(cumsum(Int.(good), dims = 1), dims = 2)
    Sp = zeros(Int, size(S) .+ 1); Sp[2:end, 2:end] = S
    boxsum(i0, j0) = Sp[i0+N+1, j0+N+1] - Sp[i0+1, j0+N+1] - Sp[i0+N+1, j0+1] + Sp[i0+1, j0+1]
    best = (0, 0); bv = -1
    for i0 in 0:stride:size(good, 1)-N, j0 in 0:stride:size(good, 2)-N
        v = boxsum(i0, j0)
        v > bv && (bv = v; best = (i0, j0))
    end
    return best, bv
end

# ---------------------------------------------------------------- Greenland
survey(GRL)
h5open(GRL, "r") do h
    x = read(h["x"]); y = read(h["y"])
    @printf("x: %d .. %d (dx %d)   y: %d .. %d (dy %d)\n", x[1], x[end], x[2] - x[1], y[1], y[end], y[2] - y[1])
    mask = read(h["mask"])
    println("mask counts 0..4: ", [count(==(v), mask) for v in 0:4])
    ice = (mask .== 2) .| (mask .== 3)
    N = 8192
    (i0, j0), nice = best_offset(ice, N)
    @printf("8192^2 crop with most ice: offset (%d, %d), ice fraction %.3f\n", i0, j0, nice / N^2)
    @printf("  crop x %.0f .. %.0f km, y %.0f .. %.0f km\n", x[i0+1] / 1e3, x[i0+N] / 1e3, y[j0+1] / 1e3, y[j0+N] / 1e3)
    I = i0+1:i0+N; J = j0+1:j0+N
    bed = h["bed"][I, J]; thk = h["thickness"][I, J]; srf = h["surface"][I, J]; msk = mask[I, J]
    errbed = h["errbed"][I, J]
    for (name, v) in (("bed", bed), ("thickness", thk), ("surface", srf))
        vv = Float64.(v)
        @printf("  %-10s min %8.1f max %8.1f mean %8.1f std %8.1f  #fill(-9999) %d\n", name, minimum(vv), maximum(vv), mean(vv), std(vv), count(==(-9999.0), vv))
    end
    icec = (msk .== 2) .| (msk .== 3)
    eb = Float64.(errbed[icec .& (errbed .> 0)])
    @printf("  errbed on ice: median %.1f m, 10%% %.1f, 90%% %.1f\n", pct(eb, 0.5), pct(eb, 0.1), pct(eb, 0.9))
    h5open(joinpath(OUT, "grl8192.h5"), "w") do o
        o["bed"] = bed; o["thickness"] = thk; o["surface"] = srf; o["mask"] = msk
        attributes(o)["x0"] = x[i0+1]; attributes(o)["y0"] = y[j0+1]; attributes(o)["dx"] = x[2] - x[1]
        attributes(o)["source"] = GRL
    end
    # whole-domain coarse overview (every 16th point) for figures
    h5open(joinpath(OUT, "grl_overview.h5"), "w") do o
        o["surface"] = h["surface"][1:16:end, 1:16:end]; o["mask"] = mask[1:16:end, 1:16:end]
        attributes(o)["crop_i0"] = i0; attributes(o)["crop_j0"] = j0; attributes(o)["stride"] = 16
    end
end

# ---------------------------------------------------------------- Antarctic velocity
survey(VEL)
h5open(VEL, "r") do h
    x = read(h["x"]); y = read(h["y"])
    @printf("x: %.0f .. %.0f (dx %.0f)   y: %.0f .. %.0f (dy %.0f)\n", x[1], x[end], x[2] - x[1], y[1], y[end], y[2] - y[1])
    N = 8192; n0 = length(x)
    i0 = (n0 - N) ÷ 2; j0 = (length(y) - N) ÷ 2
    I = i0+1:i0+N; J = j0+1:j0+N
    vx = Float64.(h["VX"][I, J]); vy = Float64.(h["VY"][I, J])
    valid = isfinite.(vx) .& isfinite.(vy) .& ((vx .!= 0) .| (vy .!= 0))
    speed = Float32.(ifelse.(valid, hypot.(vx, vy), NaN))
    sv = Float64.(speed[valid])
    @printf("crop offset (%d,%d), x %.0f .. %.0f km, valid fraction %.3f\n", i0, j0, x[i0+1] / 1e3, x[i0+N] / 1e3, mean(valid))
    @printf("speed: median %.2f, 1%% %.3f, 99%% %.1f, max %.1f m/a\n", pct(sv, 0.5), pct(sv, 0.01), pct(sv, 0.99), maximum(sv))
    for ename in ("ERRX", "STDX")
        if haskey(h, ename)
            e = Float64.(h[ename][I, J])
            ev = e[valid .& isfinite.(e) .& (e .> 0)]
            @printf("%s on valid: median %.2f, 90%% %.2f m/a\n", ename, pct(ev, 0.5), pct(ev, 0.9))
        end
    end
    h5open(joinpath(OUT, "antvel8192.h5"), "w") do o
        o["speed"] = speed
        attributes(o)["x0"] = x[i0+1]; attributes(o)["y0"] = y[j0+1]; attributes(o)["dx"] = x[2] - x[1]
        attributes(o)["source"] = VEL
    end
end

# ---------------------------------------------------------------- Antarctic bed (2 km)
survey(ANTBED)
h5open(ANTBED, "r") do h
    x = read(h["x"]); y = read(h["y"])
    @printf("x: %d .. %d (dx %d)\n", x[1], x[end], x[2] - x[1])
    N = 2048; i0 = (length(x) - N) ÷ 2; j0 = (length(y) - N) ÷ 2
    bed = h["bed"][i0+1:i0+N, j0+1:j0+N]
    bv = Float64.(bed)
    @printf("bed crop: min %.1f max %.1f std %.1f  #nonfinite %d  #(-9999) %d\n", minimum(bv), maximum(bv), std(bv), count(!isfinite, bv), count(==(-9999.0), bv))
    h5open(joinpath(OUT, "antbed2048.h5"), "w") do o
        o["bed"] = bed
        attributes(o)["x0"] = x[i0+1]; attributes(o)["y0"] = y[j0+1]; attributes(o)["dx"] = x[2] - x[1]
        attributes(o)["source"] = ANTBED
    end
end
println("done")
