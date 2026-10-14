# Benchmark of MPS linear-solver settings for the implicit SIA system (I - dt L[D]) x = H
# against a sparse direct solve (1D Halfar profile), sweeping resolution and local Krylov method.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_linsolve.jl

using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

p = SIAParams()
hb = Halfar(d = 1, p = p)
st = SolverSettings()
dt = 100.0

for nbits in (10, 12, 14)
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3))
    ops = SIAOperators(g)
    H = function_to_mps(x -> halfar_H(hb, halfar_t0(hb), abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    D = face_diffusivities(H, nothing, ops, p, st)
    A = mpo_lincomb([1.0, -dt], [ops.I, diffusion_mpo(D, ops, st)]; cutoff = st.op_cutoff)
    Hd = mps_to_array(H, g); N = length(Hd); h = g.h[1]; Dd = mps_to_array(D[1], g)
    Df = spdiagm(0 => -ones(N), 1 => ones(N - 1)) / h
    Db = spdiagm(0 => ones(N), -1 => -ones(N - 1)) / h
    As = I - dt * Db * spdiagm(0 => Dd) * Df
    tref = @elapsed xref = As \ Hd
    kappa = 1 + dt * 4 * maximum(Dd) / h^2
    println(@sprintf("nbits %d: chi(A) %d, kappa ~ %.1e, sparse direct %.3f s", nbits, maxlinkdim(A), kappa, tref))
    for (name, ukw) in [("gmres(30) tol1e-10", (; ishermitian = true, tol = 1e-10, maxiter = 30, krylovdim = 30)),
                        ("cg tol1e-8 it500", (; ishermitian = true, isposdef = true, tol = 1e-8, maxiter = 500)),
                        ("cg tol1e-10 it2000", (; ishermitian = true, isposdef = true, tol = 1e-10, maxiter = 2000))]
        for nsw in (2, 4)
            t = @elapsed x = linsolve(A, H, H; nsweeps = nsw, cutoff = st.cutoff, maxdim = st.maxdim, outputlevel = 0, updater_kwargs = ukw)
            err = maximum(abs, mps_to_array(x, g) - xref) / maximum(abs, xref)
            println(@sprintf("   %-20s nsweeps %d: %7.2f s  rel err %.2e  chi(x) %d", name, nsw, t, err, maxlinkdim(x)))
            flush(stdout)
        end
    end
end
