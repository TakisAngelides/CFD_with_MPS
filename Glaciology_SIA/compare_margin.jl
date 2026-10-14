# H in the last ice cells at the +x margin for several EISMINT runs on the same grid
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch compare_margin.jl output/a.h5 output/b.h5 ...
using HDF5, Printf
include("quantics.jl")
for f in ARGS
    Hm, nb, ord = h5open(f, "r") do io
        read(io, "H_mps", MPS), read(attributes(io)["nbits"]), Symbol(read(attributes(io)["ordering"]))
    end
    g = QuanticsGrid(nb, ord, (-750e3, 750e3))
    H = vec(mps_to_array(replace_siteinds(Hm, g.sites), g)); x = coords(g, 1)
    im = findlast(>(1e-3), H)
    println(@sprintf("%-60s dome %.3f m  ", basename(f), maximum(H)),
            join([@sprintf("%.2f km: %.3f", x[i] / 1e3, H[i]) for i in im-5:min(length(H), im + 1)], " | "))
end
