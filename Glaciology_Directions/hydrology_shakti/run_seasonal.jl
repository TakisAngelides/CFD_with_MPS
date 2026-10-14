# Seasonal channelization run (Sommers, Rajaram & Morlighem 2018, Sect. 3.3), copied from
# ~/Shakti.jl/test/channelization_seasonal/run_channelization.jl (same geometry, parameters,
# noise seed, spin-up, Picard settings), with three changes for the MPS study:
#   * grid N x N with N = 2^n (quantics needs powers of two),
#   * the run stops at END_DAY and snapshots of ALL hydrology fields (head, effective pressure,
#     gap height, face fluxes / conductivities, Reynolds number, ...) are written to HDF5 at SAVE_DAYS,
#   * at every snapshot the assembled Picard-linearised head system (sparse M, rhs) is written as well,
#     and one sparse Cholesky factorisation + solve of it is timed (reference cost for the MPS solve).
# Optional warm start (WARM = file.h5:day): bilinear interpolation of a coarser run's gap height,
# simulation clock set to that day, no spin-up.
#
# Run with the user's Shakti environment read-only (nothing in ~/Shakti.jl is written):
#   julia -t 16 --project=/albedo/home/taange001/Shakti.jl run_seasonal.jl N END_DAY SAVE_DAYS [WARM|none] [TAG] [diff]
# e.g.  run_seasonal.jl 128 210 120,170,200

using Shakti
using Random
using HDF5
using SparseArrays, LinearAlgebra
using Printf

const N = parse(Int, ARGS[1])
const END_DAY = parse(Float64, ARGS[2])
const SAVE_DAYS = parse.(Float64, split(ARGS[3], ","))
const WARM = length(ARGS) >= 4 && ARGS[4] != "none" ? ARGS[4] : nothing
const TAG = length(ARGS) >= 5 ? ARGS[5] : ""
const DIFFUSION = length(ARGS) >= 6 && ARGS[6] == "diff"   # SUHMO channel-wall diffusion of b (Felden et al. 2023, Eq. 10-11)

const NX, NY = N, N
const LX, LY = 4000.0, 8000.0
grid = Grid(NX, NY, LX, LY)

zb = zeros(NX, NY)
H0, H1 = 550.0, 700.0
Hx = sqrt.(H0^2 .+ (H1^2 - H0^2) .* (grid.x ./ LX))
zs = repeat(Hx, 1, NY)

mask = fill(GROUNDED, NX, NY)
mask[1, :]   .= LAND
mask[end, :] .= OTHER_BASIN
mask[:, 1]   .= OTHER_BASIN
mask[:, end] .= OTHER_BASIN

p = ModelParameters(e_v = 0.0, b_min = 1e-3, omega = 1e-4)

A_visc = fill(5e-25, NX, NY)
G      = fill(0.05, NX, NY)
ub_x   = fill(1e-6, NX + 1, NY)
ub_y   = zeros(NX, NY + 1)
sl     = RegularizedCoulombSlidingLaw(0.25)
taub_x = zeros(NX + 1, NY)
taub_y = zeros(NX, NY + 1)

rng = Random.MersenneTwister(1)
b = fill(0.01, NX, NY) .* (1 .+ 0.01 .* randn(rng, NX, NY))

const SECONDS_PER_YEAR = 365 * 86400.0
const DAY = 86400.0
dt = 3600.0

function bilinear(fc, xc, yc, xf, yf)
    out = zeros(length(xf), length(yf))
    for (i, x) in enumerate(xf), (j, y) in enumerate(yf)
        i0 = clamp(searchsortedlast(xc, x), 1, length(xc) - 1)
        j0 = clamp(searchsortedlast(yc, y), 1, length(yc) - 1)
        tx = clamp((x - xc[i0]) / (xc[i0+1] - xc[i0]), 0, 1)
        ty = clamp((y - yc[j0]) / (yc[j0+1] - yc[j0]), 0, 1)
        out[i, j] = (1 - tx) * (1 - ty) * fc[i0, j0] + tx * (1 - ty) * fc[i0+1, j0] +
                    (1 - tx) * ty * fc[i0, j0+1] + tx * ty * fc[i0+1, j0+1]
    end
    return out
end

start_time = 0.0
ieb_spinup = fill(1.0 / SECONDS_PER_YEAR, NX, NY)
state = State(grid)
ls = CholeskyDirectSolver(grid)
ps = PicardSolver(500, 1e-6, ls, grid; alpha = 0.1)
ds = DIFFUSION ? WithDiffusion(CholeskyDirectSolver(grid)) : NoDiffusion()

if WARM === nothing
    set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, b, G, ub_x, ub_y, ieb_spinup, taub_x, taub_y)
    sim_spinup = Simulation(grid, state, 20, floattype(dt), p, "implicit", String[], ConstantMeltInput(), sl; ps = ps, diffusion_scheme = ds)
    println("Grid: $(NX)x$(NY) -- spin-up (20 steps)")
    @time run!(sim_spinup)
else
    wfile, wday = split(WARM, ":")
    wkey = @sprintf("day_%05.1f", parse(Float64, wday))
    bc, xc, yc = h5open(wfile, "r") do f
        read(f[wkey]["b"]), read(f["x"]), read(f["y"])
    end
    bw = max.(bilinear(bc, xc, yc, grid.x, grid.y), p.b_min)
    println("Warm start from $wfile [$wkey]: b in ", extrema(bw))
    set_initial_conditions!(state, grid, p, sl, mask, A_visc, zb, zs, bw, G, ub_x, ub_y, ieb_spinup, taub_x, taub_y)
    start_time = parse(Float64, wday) * DAY
end

tsteps = round(Int, (END_DAY * DAY - start_time) / dt)
sim = Simulation(grid, state, tsteps, floattype(dt), p, "implicit", String[], SeasonalMeltInput(), sl; ps = ps, diffusion_scheme = ds)
sim.total_time[] = start_time

outdir = joinpath(@__DIR__, "data")
mkpath(outdir)
outpath = joinpath(outdir, "seasonal_N$(N)$(TAG).h5")
h5open(outpath, "w") do f
    f["x"] = collect(grid.x); f["y"] = collect(grid.y)
    f["dx"] = grid.dx; f["dy"] = grid.dy
    f["mask"] = Array(state.mask)
end

const FIELDS = [:h, :pw, :po, :N, :b, :mdot, :ieb, :K, :Re, :zb, :zs, :beta, :lc, :abs_ub, :A_visc,
                :q_x, :q_y, :K_x, :K_y, :Re_x, :Re_y, :dhdx, :dhdy]

function save_snapshot(day, iters, times)
    key = @sprintf("day_%05.1f", day)
    s = state
    # assemble the Picard-linearised head system at this state (the first system of the next step)
    Shakti.update_SALS_elliptic!(ls.sals, s, grid, p, sim.kfs, sim.ds)
    M = copy(ls.sals.M); rhs = copy(ls.sals.rhs)
    I, J, V = findnz(M)
    t_fact = @elapsed F = cholesky(Symmetric(M))
    t_fact2 = @elapsed F = cholesky(Symmetric(M))
    t_solve = @elapsed hsol = F \ rhs
    t_solve2 = @elapsed hsol = F \ rhs
    relres = norm(M * hsol - rhs) / norm(rhs)
    dh = maximum(abs, hsol .- vec(Array(s.h)))
    h5open(outpath, "r+") do f
        g = create_group(f, key)
        for fld in FIELDS
            g[string(fld)] = Array(getfield(s, fld))
        end
        g["M_I"] = I; g["M_J"] = J; g["M_V"] = V; g["rhs"] = rhs; g["h_sparse"] = hsol
        g["t_chol"] = min(t_fact, t_fact2); g["t_solve"] = min(t_solve, t_solve2)
        g["time"] = sim.total_time[]
        g["picard_iters"] = iters; g["step_times"] = times
    end
    @printf("saved %s: b in [%.3e, %.3e], N in [%.3e, %.3e], chol %.3fs solve %.3fs, relres %.1e, |h_sparse - h| = %.2e m\n",
            key, extrema(Array(s.b))..., extrema(Array(s.N))..., min(t_fact, t_fact2), min(t_solve, t_solve2), relres, dh)
    flush(stdout)
end

save_steps = Dict(round(Int, (d * DAY - start_time) / dt) => d for d in SAVE_DAYS if d * DAY >= start_time)
iters = Int[]; times = Float64[]
println("Grid $(NX)x$(NY), dx=$(grid.dx) m, dy=$(grid.dy) m, steps $tsteps from day $(start_time / DAY) to $END_DAY, saves at $(sort(collect(values(save_steps))))")
flush(stdout)
0 in keys(save_steps) && save_snapshot(save_steps[0], iters, times)
t_total = @elapsed for t in 1:tsteps
    st = @elapsed Shakti.step!(sim)
    sim.total_time[] += sim.dt[]
    conv, it = Shakti.picard_status(sim.hs)
    push!(iters, it); push!(times, st)
    if t % 24 == 0 || !conv
        @printf("day %.2f: step %.3fs, Picard %s %d it, b max %.4f, N min %.3e\n", sim.total_time[] / DAY, st, conv, it,
                maximum(Array(state.b)), minimum(Array(state.N)))
        flush(stdout)
    end
    haskey(save_steps, t) && save_snapshot(save_steps[t], iters, times)
end
@printf("done: %d steps in %.1f s, %d Picard iterations total (%.4f s/iteration)\n", tsteps, t_total, sum(iters), t_total / max(sum(iters), 1))
