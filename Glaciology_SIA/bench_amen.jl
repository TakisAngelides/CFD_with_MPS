# AMEn (amen.jl) vs ITensorMPS two-site linsolve on a 2D variable-coefficient diffusion system
#   (I - c div(k grad)) x = b on a quantics grid, interleaved, Neumann-ish open shifts.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_amen.jl bits=6,8 c=1e-2
using Printf
include("quantics.jl"); include("sia.jl"); include("amen.jl")

opts = Dict("bits" => "6,8", "c" => "1e-2", "maxdim" => "48", "sweeps" => "8")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
c = parse(Float64, opts["c"]); md = parse(Int, opts["maxdim"]); nsw = parse(Int, opts["sweeps"])
for nb in parse.(Int, split(opts["bits"], ","))
    g = QuanticsGrid(nb, :interleaved, (0.0, 1.0), (0.0, 1.0)); N = 2^nb
    Id = identity_mpo(g)
    kf = function_to_mps((x, y) -> 1 + 0.5sin(3x) * cos(2y) + x * y, g; tolerance = 1e-12)
    K = diag_mpo(kf)
    L = nothing
    for a in 1:2
        Sp = shift_mpo(g, a, +1); Sm = shift_mpo(g, a, -1)
        D = mpo_lincomb([1.0, -1.0], [Sp, Id])           # forward difference (open)
        Dm = mpo_lincomb([1.0, -1.0], [Id, Sm])           # backward difference
        La = mpo_product(Dm, mpo_product(K, D))
        L = L === nothing ? La : mpo_lincomb([1.0, 1.0], [L, La])
    end
    A = mpo_lincomb([1.0, -c * N^2], [Id, L]; cutoff = 1e-14)
    b = function_to_mps((x, y) -> exp(-20((x - 0.3)^2 + (y - 0.6)^2)) + x, g; tolerance = 1e-12)
    @printf("2^%d x 2^%d: chi(A) %d chi(b) %d, c N^2 = %.1f\n", nb, nb, maxlinkdim(A), maxlinkdim(b), c * N^2)
    t1 = @elapsed (x1, i1) = amen_solve(A, b, b; nsweeps = nsw, tol = 1e-10, cutoff = 1e-10, maxdim = md, kick = 4, verbose = true)
    @printf("  amen:     res %.2e  sweeps %d  chi %d  %.2f s\n", i1[1], i1[2], maxlinkdim(x1), t1)
    t1 = @elapsed (x1, i1) = amen_solve(A, b, b; nsweeps = nsw, tol = 1e-10, cutoff = 1e-10, maxdim = md, kick = 4)
    @printf("  amen(2nd):res %.2e  sweeps %d  chi %d  %.2f s\n", i1[1], i1[2], maxlinkdim(x1), t1)
    for ns in (2, 4, nsw)
        t2 = @elapsed x2 = linsolve(A, b, b; nsweeps = ns, cutoff = 1e-10, maxdim = md, outputlevel = 0, updater = direct_updater,
                                    updater_kwargs = (; maxdirect = 20000))
        @printf("  dmrg %2d:  res %.2e  chi %d  %.2f s\n", ns, residual_norm(A, x2, b), maxlinkdim(x2), t2)
    end
    flush(stdout)
end
