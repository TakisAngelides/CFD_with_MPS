# QTT rank / compression analysis of Shakti hydrology fields, with an AMR (quadtree) comparison.
#   julia --project=~/CFD_with_MPS rank_analysis.jl OUT.txt file1.h5:day1,day2 file2.h5:day ...
# For every snapshot and field (head h, effective pressure N, gap height b, log10 b, |q|, K):
#   * exact bond spectra (TT-SVD) for :interleaved and :sequential orderings -> eps-ranks
#   * truncated QTT at a ladder of relative Frobenius tolerances: params, max rank, errors
#   * quadtree with bilinear leaves at a ladder of max-error tolerances: leaves, nodes
#   * gradient tagging (AMR refinement flag) at tau = 1e-2, 5e-2
# Results are appended as whitespace-separated rows to OUT.txt (one row per measurement).

using HDF5, Printf, LinearAlgebra
include(joinpath(@__DIR__, "qtt_tools.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))

const OUT = ARGS[1]
function cell_qmag(qx, qy)
    n = size(qx, 2)
    qcx = (qx[1:n, :] .+ qx[2:n+1, :]) ./ 2
    qcy = (qy[:, 1:n] .+ qy[:, 2:n+1]) ./ 2
    return sqrt.(qcx .^ 2 .+ qcy .^ 2)
end

# The one-cell rim of non-GROUNDED cells (LAND outlet column x=1, OTHER_BASIN at the other three
# edges) holds solver bookkeeping values (b = 0, frozen initial head, ...). Replace them by the
# adjacent interior value so that only the physical field is measured (the outlet head h = 0 is kept).
function fill_rim(A, mask; keep_land = false)
    B = copy(A); n = size(A, 1)
    for j in 1:n, i in 1:n
        mask[i, j] == 0 && continue
        keep_land && mask[i, j] == 2 && continue
        B[i, j] = A[clamp(i, 2, n - 1), clamp(j, 2, n - 1)]
    end
    return B
end

function snapshot_fields(file, day)
    key = @sprintf("day_%05.1f", day)
    h5open(file, "r") do f
        mask = read(f["mask"])
        g = f[key]
        b = fill_rim(read(g["b"]), mask)
        Dict("h" => fill_rim(read(g["h"]), mask; keep_land = true), "N" => fill_rim(read(g["N"]), mask), "b" => b,
             "logb" => log10.(max.(b, 1e-3)), "qmag" => fill_rim(cell_qmag(read(g["q_x"]), read(g["q_y"])), mask),
             "K" => fill_rim(read(g["K"]), mask))
    end
end

io = open(OUT, "a")

# sanity: orderings / reconstruction round trip on a random 2^5 x 2^5 array
let A = randn(32, 32)
    for ord in (:interleaved, :sequential)
        tt, _ = tt_svd(to_tensor(A, ord))
        err = maximum(abs, from_tensor(tt_full(tt), 5, ord) - A)
        @printf("round trip %s: %.2e\n", ord, err)
        @assert err < 1e-10
    end
    # interleaved point check: site k of x is bit k (MSB first) of i-1
    T = to_tensor(A, :interleaved)
    i, j = 19, 7
    bx = digits(i - 1; base = 2, pad = 5) |> reverse; by = digits(j - 1; base = 2, pad = 5) |> reverse
    idx = vcat([[bx[k] + 1, by[k] + 1] for k in 1:5]...)
    @assert T[idx...] == A[i, j]
    println("point check ok")
end

for spec in ARGS[2:end]
    file, days = split(spec, ":")
    N = h5open(f -> length(read(f["x"])), file, "r")
    for d in parse.(Float64, split(days, ","))
        flds = snapshot_fields(file, d)
        label = @sprintf("N%d_day%03d%s", N, round(Int, d), occursin("diff", file) ? "_diff" : "")
        for name in ("h", "N", "b", "logb", "qmag", "K")
            t = @elapsed analyze(io, label, name, flds[name])
            @printf("%s %s done in %.1fs\n", label, name, t)
        end
    end
end
close(io)
