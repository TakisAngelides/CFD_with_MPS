# Rank vs resolution for synthetic channel networks with a controlled channel width.
#   julia --project=~/CFD_with_MPS synthetic_ranks.jl OUT.txt nmin nmax [amr=1]
# Domain 4 km x 8 km (as the Shakti seasonal test), 2^n x 2^n cells. Gap-height-like field
#   b = b_bg(x) + sum_c b_c(x) exp(-(d_c / w)^2),   d_c = distance to channel centreline c
# Networks:
#   "dendritic": 6 x-aligned trunks near the outlet, each splitting twice into curved, meandering
#                branches (Shakti-like); total centreline length ~ 40 km
#   "straight":  6 x-aligned trunks over the whole length (best case for QTT)
# Widths: fixed physical w = 25 m and 8 m ("phys25", "phys8"), and grid-scale w = 1.5 dx ("grid").

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "qtt_tools.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "4")))

const OUT = ARGS[1]
const NMIN = parse(Int, ARGS[2]); const NMAX = parse(Int, ARGS[3])
const AMR = length(ARGS) >= 4 ? ARGS[4] == "1" : true
const LX, LY = 4000.0, 8000.0

# channel = (centreline y = f(x), slope f'(x), x range, amplitude)
struct Channel
    f::Function
    fp::Function
    x0::Float64
    x1::Float64
    amp::Float64
end

function network(kind)
    ch = Channel[]
    ys = [(k - 0.5) * LY / 6 + 90 * sin(1.7k) for k in 1:6]
    for (k, y0) in enumerate(ys)
        if kind == :straight
            push!(ch, Channel(x -> y0, x -> 0.0, 0.0, 3800.0, 0.12))
            continue
        end
        push!(ch, Channel(x -> y0, x -> 0.0, 0.0, 1300.0, 0.12))                       # trunk
        for s in (-1, 1)
            A1 = 260.0 * (1 + 0.15 * sin(3.1k + s))
            ph = 2.3k + 1.1s
            f1(x) = y0 + s * A1 * (1 - cos(pi * min(x - 1300, 1500) / 1500)) / 2 + 25 * sin(2pi * (x - 1300) / 650 + ph) * min((x - 1300) / 300, 1)
            f1p(x) = s * A1 * (x - 1300 < 1500 ? pi / 1500 * sin(pi * (x - 1300) / 1500) / 2 : 0.0) +
                     25 * 2pi / 650 * cos(2pi * (x - 1300) / 650 + ph) * min((x - 1300) / 300, 1) +
                     (x - 1300 < 300 ? 25 / 300 * sin(2pi * (x - 1300) / 650 + ph) : 0.0)
            push!(ch, Channel(f1, f1p, 1300.0, 3700.0, 0.07))                                # first-order branch
            for s2 in (-1, 1)
                A2 = 110.0 * (1 + 0.2 * cos(1.3k + s + s2))
                f2(x) = f1(x) + s2 * A2 * (1 - cos(pi * min(x - 2300, 1300) / 1300)) / 2
                f2p(x) = f1p(x) + s2 * A2 * (x - 2300 < 1300 ? pi / 1300 * sin(pi * (x - 2300) / 1300) / 2 : 0.0)
                push!(ch, Channel(f2, f2p, 2300.0, 3900.0, 0.035))                          # second-order branch
            end
        end
    end
    return ch
end

function field(n, kind, w_of_dx)
    N = 2^n
    dx, dy = LX / N, LY / N
    w = w_of_dx(dx)
    xs = [(i - 0.5) * dx for i in 1:N]; ys = [(j - 0.5) * dy for j in 1:N]
    B = [1e-3 + 2e-3 * x / LX for x in xs, y in ys]
    for c in network(kind)
        for (i, x) in enumerate(xs)
            (x < c.x0 - 3w || x > c.x1 + 3w) && continue
            xc = clamp(x, c.x0, c.x1)
            yc = c.f(xc); sl = c.fp(xc)
            taper = x < c.x0 ? exp(-((c.x0 - x) / w)^2) : x > c.x1 ? exp(-((x - c.x1) / w)^2) : 1.0
            cosang = 1 / sqrt(1 + sl^2)
            # only rows within 6 w / cos of the centreline
            jlo = max(1, floor(Int, (yc - 6w / cosang) / dy)); jhi = min(N, ceil(Int, (yc + 6w / cosang) / dy) + 1)
            for j in jlo:jhi
                d = (ys[j] - yc) * cosang
                B[i, j] += c.amp * taper * exp(-(d / w)^2)
            end
        end
    end
    return B, w
end

io = open(OUT, "a")
for n in NMIN:NMAX
    for (kind, wname, wfun) in ((:dendritic, "phys25", dx -> 25.0), (:dendritic, "phys8", dx -> 8.0),
                                (:dendritic, "grid", dx -> 1.5dx), (:straight, "grid", dx -> 1.5dx))
        B, w = field(n, kind, wfun)
        label = "syn_$(kind)_$(wname)"
        t = @elapsed analyze(io, label, "b", B; amr = AMR, qtt = n <= 11)
        @printf("%s n=%d (dx=%.2f m, w=%.2f m) done in %.1fs\n", label, n, LX / 2^n, w, t)
    end
end
close(io)
