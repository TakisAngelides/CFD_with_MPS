# Run an SIA benchmark with the quantics-MPS solver.
#
#   julia --project=.. run_sia.jl                                   # defaults below
#   julia --project=.. run_sia.jl experiment=eismint nbits=9 dt=200 ordering=sequential
#
# Any key of `cfg` can be overridden as key=value on the command line.

using HDF5
using Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

cfg = Dict{String,Any}(
    "experiment"   => "halfar",       # "halfar" (Bueler Test B) or "eismint" (EISMINT I moving margin)
    "d"            => 2,              # 1 (flowline) or 2 (map plane)
    "nbits"        => 8,              # 2^nbits points per dimension
    "ordering"     => "interleaved",  # "interleaved" or "sequential"
    "dt"           => 100.0,          # years (largest step of the adaptive stepping)
    "dt_start"     => 0.0,            # first step (> 0: start smaller and grow by dt_grow, e.g. after a restart)
    "t_run"        => 25000.0,        # years of simulation (Halfar: starting at t0)
    "maxdim"       => 128,
    "cutoff"       => 1e-20,          # state truncation; the residual noise floor ~ H (D/h^2) dt sqrt(cutoff) hides tendencies below it
    "op_maxdim"    => 128,
    "op_cutoff"    => 1e-22,
    "scheme"       => "bdf2",         # "be" (backward Euler) or "bdf2" (second order, first step BE)
    "nonlinear"    => "newton",       # "newton" (Picard first iterate + Newton) or "picard"
    "max_iters"    => 10,             # nonlinear iterations per step
    "picard_first" => false,          # Newton: one Picard iterate first
    "extrapolate"  => false,          # initial guess 2 H^k - H^(k-1)
    "newton_log"   => false,          # print every nonlinear iteration
    "positivity"   => true,           # EISMINT: H <- max(H, 0) after each step (TCI)
    "positivity_tol" => 1e-8,         # TCI tolerance of max(H, 0); its noise ~ tol max(H) is amplified by dt D/h^2
    "complementarity" => true,        # EISMINT: H >= 0 as a complementarity problem (semismooth Newton)
    "ncp_exact_mask" => false,        # debug, small grids: active-set mask from dense arrays
    "ncp_mask_alpha" => 0.0,          # EISMINT: > 0 smooth TCI mask 1 - exp(-alpha (q + |q|)), q = H - c R (1/m)
    "ncp_scale"    => 1.0,            # EISMINT: c in min(H, c R) (active-set guess; c < 1 favours the flow equation)
    "limit_ablation" => false,        # EISMINT: (dt-dependent) ablation limited by the available ice
    "tol"          => 1e-5,           # nonlinear tolerance ||dH||/||H||
    "noise_tol"    => 1e-4,           # accepted ||dH||/||H|| when the residual is at its noise floor
    "max_halvings" => 8,              # adaptive dt: smallest step is dt / 2^max_halvings
    "dt_grow"      => 1.5,            # adaptive dt: growth factor after an easy step
    "grow_iters"   => 6,              # adaptive dt: a step is easy if Newton needed <= grow_iters iterations
    "nsweeps"      => 2,
    "prod_cutoff"  => 1e-24,          # truncation inside the nonlinear products (see SolverSettings)
    "prod_maxdim"  => 256,
    "product_alg"  => "naive",        # MPO x MPS products: "naive" (exact + SVD) or "densitymatrix"
    "linear_solver" => "linsolve",    # "linsolve" or "inverse" (tracked Newton-Schulz MPO inverse)
    "local_solver" => "direct",       # linsolve local problems: "direct" (dense LU) or "krylov"
    "direct_maxdim" => 6000,          # largest local problem solved by LU
    "lin_maxdim"   => 32,             # max bond dimension of the Newton/Picard update
    "lin_cutoff"   => 1e-12,          # truncation of the Newton/Picard update
    "lin_retries"  => 3,
    "lin_refine"   => 0,
    "lin_tol"      => 0.0,            # > 0: sweep the linear solve to this exact relative residual
    "lin_maxsweeps" => 64,
    "lin_refine_tol" => 1e-6,
    "lin_res_tol"  => 0.1,            # re-sweep the linear solve while its relative residual is above this
    "inv_maxdim"   => 256,
    "tci_tol"      => 1e-8,
    "output_every" => 10,             # steps between snapshots
    "sample"       => 256,            # snapshots/errors on a sample^d sub-lattice (<= 2^nbits)
    "outdir"       => joinpath(@__DIR__, "output"),
    "tag"          => "",             # appended to the output file name
    "restart"      => "",             # .h5 of an earlier run (same experiment and d, any nbits/ordering):
                                      # its final H, interpolated to this grid, is the initial state
)

for arg in ARGS
    key, val = split(arg, "=", limit = 2)
    haskey(cfg, key) || error("unknown option $(key), valid: $(sort(collect(keys(cfg))))")
    cfg[key] = cfg[key] isa String ? String(val) : parse(typeof(cfg[key]), val)
end

d, nbits = cfg["d"], cfg["nbits"]
ordering = Symbol(cfg["ordering"])
p = SIAParams()
PRODUCT_ALG[] = cfg["product_alg"]
st = SolverSettings(cutoff = cfg["cutoff"], maxdim = cfg["maxdim"], op_cutoff = cfg["op_cutoff"],
                    prod_cutoff = cfg["prod_cutoff"], prod_maxdim = cfg["prod_maxdim"],
                    local_solver = Symbol(cfg["local_solver"]), direct_maxdim = cfg["direct_maxdim"], lin_maxdim = cfg["lin_maxdim"], lin_cutoff = cfg["lin_cutoff"],
                    picard_first = cfg["picard_first"], extrapolate = cfg["extrapolate"], newton_log = cfg["newton_log"],
                    op_maxdim = cfg["op_maxdim"], nonlinear = Symbol(cfg["nonlinear"]),
                    max_iters = cfg["max_iters"], tol = cfg["tol"], noise_tol = cfg["noise_tol"], nsweeps = cfg["nsweeps"],
                    linear_solver = Symbol(cfg["linear_solver"]), lin_retries = cfg["lin_retries"], lin_refine = cfg["lin_refine"], lin_tol = cfg["lin_tol"], lin_maxsweeps = cfg["lin_maxsweeps"], lin_refine_tol = cfg["lin_refine_tol"], lin_res_tol = cfg["lin_res_tol"], inv_maxdim = cfg["inv_maxdim"],
                    positivity = cfg["experiment"] == "eismint" && cfg["positivity"], positivity_tol = cfg["positivity_tol"],
                    limit_ablation = cfg["experiment"] == "eismint" && cfg["limit_ablation"],
                    complementarity = cfg["experiment"] == "eismint" && cfg["complementarity"], ncp_scale = cfg["ncp_scale"], ncp_exact_mask = cfg["ncp_exact_mask"], ncp_mask_alpha = cfg["ncp_mask_alpha"])

radius(x...) = sqrt(sum(abs2, x))

if cfg["experiment"] == "halfar"
    bench = Halfar(d = d, p = p)
    L = 1200e3
    t_start = halfar_t0(bench)
    exact(t, x...) = halfar_H(bench, t, radius(x...))
    a_fun = nothing
elseif cfg["experiment"] == "eismint"
    bench = EISMINTMovingMargin(p = p)
    L = bench.L
    t_start = 0.0
    exact = nothing
    a_fun = (x...) -> eismint_mb(bench, radius(x...))
else
    error("unknown experiment $(cfg["experiment"])")
end

g = QuanticsGrid(nbits, ordering, ntuple(_ -> (-L, L), d)...)
ops = SIAOperators(g)
println(@sprintf("%s: d = %d, %d^%d grid (dx = %.2f km), %d sites, ordering = %s",
                 cfg["experiment"], d, npoints(g), d, g.h[1] / 1e3, nsites(g), ordering)); flush(stdout)

# Initial state and forcing
a_mb = a_fun === nothing ? nothing : function_to_mps(a_fun, g; tolerance = cfg["tci_tol"], maxbonddim = cfg["maxdim"])
if !isempty(cfg["restart"])
    # grid sequencing: final state of an earlier run, interpolated to this grid
    Hc, nb_c, ord_c, t_c = h5open(cfg["restart"], "r") do f
        read(f, "H_mps", MPS), read(attributes(f)["nbits"]), Symbol(read(attributes(f)["ordering"])), read(f["times"])[end]
    end
    gc = QuanticsGrid(nb_c, ord_c, ntuple(_ -> (-L, L), d)...)
    # exact MPS prolongation when the grids nest (same ordering, finer); TCI otherwise, whose noise
    # (~ tci_tol max H) the first steps on the fine grid see amplified by dt D/h^2
    exact_transfer = ord_c == ordering && nb_c <= nbits
    H = exact_transfer ? prolong_mps(Hc, gc, g; cutoff = cfg["cutoff"], maxdim = cfg["maxdim"]) :
                         interpolate_mps(Hc, gc, g; tolerance = cfg["tci_tol"], maxbonddim = cfg["maxdim"])
    global t_start = t_c
    println(@sprintf("restart from %s: 2^%d %s grid at t = %.1f a (%s transfer)", basename(cfg["restart"]), nb_c, ord_c, t_c,
                     exact_transfer ? "exact MPS" : "TCI"))
elseif exact !== nothing
    H = function_to_mps((x...) -> exact(t_start, x...), g; tolerance = cfg["tci_tol"], maxbonddim = cfg["maxdim"])
else
    H = function_to_mps((x...) -> max(a_fun(x...), 0.0) * cfg["dt"], g; tolerance = cfg["tci_tol"], maxbonddim = cfg["maxdim"])
end
# nonzeros per row of the equivalent sparse operator (for the compression of the operator MPO)
op_stencil = cfg["nonlinear"] == "newton" ? (d == 1 ? 3 : 9) : 2d + 1
ngrid = 2.0^nsites(g)
println(@sprintf("initial H: chi %d, %d numbers stored for %.4g grid values -> compression x%.3g", maxlinkdim(H), nparams(H), ngrid, compression(H, g)))
a_mb === nothing || println(@sprintf("mass balance a: chi %d, %d numbers -> compression x%.3g", maxlinkdim(a_mb), nparams(a_mb), compression(a_mb, g)))

dt_expl, Dmax = explicit_dt_limit(H, nothing, g, ops, p, st)
println(@sprintf("max D = %.3e m^2/a, explicit limit dt ~ %.3e a, chosen dt = %.1f a (%.0fx)",
                 Dmax, dt_expl, cfg["dt"], cfg["dt"] / dt_expl))

# Sub-lattice for errors and snapshots
m = min(cfg["sample"], npoints(g))
sidx = round.(Int, range(0, npoints(g) - 1, length = m))
scoords = [[coord(g, a, i) for i in sidx] for a in 1:d]
center_bits = idx_to_bits(g, ntuple(_ -> npoints(g) ÷ 2, d))

function diagnostics(H, t)
    _, Hs = sample_grid(H, g, m)
    vol = mps_sum(g, H) * prod(g.h)
    Hc = MPSEvaluator(H)(center_bits)
    res = Dict{String,Float64}("volume" => vol, "H_center" => Hc, "H_max" => maximum(Hs))
    # margin along +x through the centre on the sub-lattice (resolution 2L/sample)
    line = d == 1 ? vec(Hs) : Hs[:, (m + 1) ÷ 2]
    ip = findlast(>(1.0), line)
    res["margin_sampled"] = ip === nothing ? 0.0 : scoords[1][ip]
    if exact !== nothing
        He = [exact(t, ntuple(a -> scoords[a][I[a]], d)...) for I in CartesianIndices(Hs)]
        err = abs.(Hs .- He)
        res["err_max"] = maximum(err)
        res["err_mean_ice"] = sum(err[He .> 0]) / count(He .> 0)
        res["err_center"] = abs(Hc - exact(t, ntuple(a -> coord(g, a, npoints(g) ÷ 2), d)...))
        res["volume_exact"] = halfar_volume(bench, t)
        res["err_volume_rel"] = abs(vol - res["volume_exact"]) / res["volume_exact"]
    end
    return res, Hs
end

times = Float64[t_start]
snap_times = Float64[t_start]
diag0, Hs0 = diagnostics(H, t_start)
diag_hist = [diag0]
snaps = [Hs0]
step_info = NamedTuple[]
substeps = Int[]       # rejected attempts before each accepted step
dts = Float64[]
walltimes = Float64[]

# Output, written at every snapshot (so a run that hits the time limit is not lost) and at the end
mkpath(cfg["outdir"])
fname = joinpath(cfg["outdir"], @sprintf("%s_d%d_n%d_%s_%s_%s_%s_dt%g_chi%d%s.h5", cfg["experiment"], d, nbits, ordering, cfg["scheme"],
                                         cfg["nonlinear"], cfg["linear_solver"], cfg["dt"], cfg["maxdim"], isempty(cfg["tag"]) ? "" : "_" * cfg["tag"]))
function write_output(H, final)
    # velocity field (cell-centred, m/a) on the sub-lattice
    uvel = velocity(H, nothing, ops, p, st)
    speed_s = sqrt.(sum(abs2.(sample_grid(u, g, m)[2]) for u in uvel))
    margin_num = margin_exact = NaN
    if exact === nothing
        margin_exact = eismint_margin(bench, d)
        # margin = largest x along the +x axis (through the centre) where H > 1 m
        ev = MPSEvaluator(H)
        line = [ev(idx_to_bits(g, ntuple(a -> a == 1 ? i : npoints(g) ÷ 2, d))) for i in 0:npoints(g)-1]
        imax = findlast(>(1.0), line)
        margin_num = imax === nothing ? 0.0 : coord(g, 1, imax - 1) + g.h[1] / 2
        final && println(@sprintf("EISMINT margin: model %.1f km, mass-balance closure %.1f km", margin_num / 1e3, margin_exact / 1e3))
    end
    h5open(fname, "w") do f
        for (k, v) in cfg
            attributes(f)[k] = v isa Symbol ? String(v) : v
        end
        attributes(f)["finished"] = final
        f["times"] = times
        f["walltime"] = walltimes
        f["chi_H"] = [s.H_maxdim for s in step_info]
        f["chi_A"] = [s.op_maxdim for s in step_info]
        f["chi_X"] = [s.X_maxdim for s in step_info]
        # storage: numbers held by H and by the operator MPO; compression vs N^d values / sparse nnz
        f["params_H"] = [s.H_params for s in step_info]
        f["params_A"] = [s.op_params for s in step_info]
        attributes(f)["ngrid"] = ngrid
        attributes(f)["op_stencil"] = op_stencil
        f["substeps"] = substeps
        f["dt"] = dts
        f["ns_iters"] = [s.ns_iters for s in step_info]
        f["nonlinear_iters"] = [s.iters for s in step_info]
        f["nonlinear_change"] = [s.change for s in step_info]
        for key in keys(diag_hist[end])
            f["diag/$(key)"] = [dg[key] for dg in diag_hist]
        end
        f["snap_times"] = snap_times
        f["snapshots"] = cat(snaps...; dims = d + 1)
        for a in 1:d
            f["coords/x$(a)"] = scoords[a]
        end
        f["speed_final"] = speed_s
        f["linkdims_final"] = linkdims(H)
        f["margin_model"] = margin_num
        f["margin_exact"] = margin_exact
        write(f, "H_mps", H)
    end
    final && println("wrote ", fname)
end

# Adaptive time stepping. The scheme is A-stable for any dt, but each implicit step must also be
# solvable: Newton advances a degenerate margin only about one cell per iteration, so a step in which
# the margin would cross many cells fails. A failed step is retried with dt/2 (variable-step BDF2),
# after easy steps dt grows again (x dt_grow) up to the requested dt.
t = t_start
t_end = t_start + cfg["t_run"]
H_prev = nothing
dt_prev = nothing
dt_cur = cfg["dt_start"] > 0 ? min(cfg["dt_start"], cfg["dt"]) : cfg["dt"]
dt_min = cfg["dt"] / 2^cfg["max_halvings"]
tracker = InverseTracker()
nstep = 0
while t < t_end - 1e-6 * cfg["dt"]
    global H, H_prev, t, dt_prev, dt_cur, nstep
    rejected = 0
    wt = 0.0
    local H_new, info, dt_try
    while true
        dt_try = min(dt_cur, t_end - t)
        H_hist = cfg["scheme"] == "bdf2" ? H_prev : nothing
        omega = dt_prev === nothing ? 1.0 : dt_try / dt_prev
        wt += @elapsed H_new, info = implicit_step(H, H_hist, dt_try, a_mb, nothing, g, ops, p, st; tracker = tracker, omega = omega)
        (info.converged || dt_try <= dt_min * (1 + 1e-9)) && break
        rejected += 1
        dt_cur = dt_try / 2
        println(@sprintf("  step rejected (newton %d, |dH|/|H| %.1e), dt -> %.4g a", info.iters, info.change, dt_cur)); flush(stdout)
    end
    nstep += 1
    t += dt_try
    H_prev, H = H, H_new
    dt_prev = dt_try
    info.converged && info.iters <= cfg["grow_iters"] && (dt_cur = min(cfg["dt"], dt_cur * cfg["dt_grow"]))
    push!(walltimes, wt); push!(step_info, info); push!(times, t); push!(substeps, rejected); push!(dts, dt_try)
    dg, Hs = diagnostics(H, t)
    push!(diag_hist, dg)
    final = t >= t_end - 1e-6 * cfg["dt"]
    if nstep % cfg["output_every"] == 0 || final
        push!(snaps, Hs); push!(snap_times, t)
        write_output(H, false)
    end
    errstr = haskey(dg, "err_max") ? @sprintf(" err_max %.2f m  err_ctr %.2f m  dV/V %.2e", dg["err_max"], dg["err_center"], dg["err_volume_rel"]) :
                                     @sprintf(" margin ~%.0f km", dg["margin_sampled"] / 1e3)
    invstr = st.linear_solver == :inverse ? @sprintf("  NS %d chi(X) %d res %.1e", info.ns_iters, info.X_maxdim, info.inv_res) : ""
    println(@sprintf("step %4d t = %9.1f a  dt %7.3g%s  H_ctr %7.1f m  chi(H) %3d (x%.3g)  chi(A) %3d (x%.3g)  %s %d (%.1e)%s res %.0e  %.1f s [D %.1f op %.1f solve %.1f]%s%s",
                     nstep, t, dt_try, rejected > 0 ? @sprintf(" (%d rej)", rejected) : "", dg["H_center"],
                     info.H_maxdim, ngrid / info.H_params, info.op_maxdim, op_stencil * ngrid / max(info.op_params, 1),
                     cfg["nonlinear"], info.iters, info.change, info.converged ? (info.noise_accept ? " noise-floor" : "") : " NOT CONVERGED", info.lin_res[1],
                     wt, info.t_D, info.t_op, info.t_solve, invstr, errstr))
    flush(stdout)
end

write_output(H, true)
