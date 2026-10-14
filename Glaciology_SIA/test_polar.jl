# Polar SIA operator: with D = 1 it must be the polar Laplacian (second order, interior points)
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch test_polar.jl
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl"); include("polar.jl")
check(name, err, tol) = println(@sprintf("%-60s %.2e  %s", name, err, err < tol ? "ok" : "FAIL (tol $(tol))"))
st = SolverSettings(cutoff = 1e-15, maxdim = 1024, op_maxdim = 4096)
for nb in (6, 8)
    g = QuanticsGrid(nb, :interleaved, (0.0, 1.0), (0.0, 2pi)); po = PolarOperators(g)
    one = constant_mps(g, 1.0)
    L = polar_diffusion_mpo([one, one], po, st)
    rs, ths = coords(g, 1), coords(g, 2)
    inner = [0.1 < r < 0.8 for r in rs, th in ths]
    for (name, f, lap) in (("r^2", (r, th) -> r^2, (r, th) -> 4.0),
                           ("r^2 cos 2theta (harmonic)", (r, th) -> r^2 * cos(2th), (r, th) -> 0.0),
                           ("r^3 sin theta", (r, th) -> r^3 * sin(th), (r, th) -> 8r * sin(th)),
                           ("exp(-4 r^2)", (r, th) -> exp(-4r^2), (r, th) -> (64r^2 - 16) * exp(-4r^2)))
        u = function_to_mps(f, g; tolerance = 1e-14, maxbonddim = 64)
        Lu = mps_to_array(mps_apply(L, u; cutoff = 1e-20), g)
        ref = [lap(r, th) for r in rs, th in ths]
        err = maximum(abs.(Lu - ref)[inner]) / max(1.0, maximum(abs, ref[inner]))
        check("2^$nb polar Laplacian of $name (interior, rel)", err, nb == 6 ? 5e-2 : 5e-3)
    end
    println("chi(L) = ", maxlinkdim(L))
end
