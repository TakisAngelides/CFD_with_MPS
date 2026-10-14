using HDF5, Printf
for f in filter(x -> occursin("dense1d_eismint", x), readdir("output"; join = true))
    h5open(f, "r") do io
        x = read(io, "x"); H = read(io, "H_final"); t = read(io, "times")[end]
        i = findlast(>(1.0), H)
        h = x[2] - x[1]
        println(@sprintf("%s: t = %.0f a, dome %.1f m, margin %.2f km (last cell with H > 1 m; h = %.3f km)",
                         basename(f), t, maximum(H), (x[i] + h / 2) / 1e3, h / 1e3))
    end
end
