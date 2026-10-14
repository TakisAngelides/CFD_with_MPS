# Tracked Newton-Schulz MPO inverse vs linsolve (CG) vs sparse direct for (I - dt L[D]) x = H
# (Halfar profiles), cold start and warm start (operator of the next time step), 1D and 2D.
#   sbatch --time=03:00:00 slurm/run.sbatch bench_inverse.jl [maxbits1d=16] [maxbits2d=9]

using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("maxbits1d" => 16, "maxbits2d" => 9, "linsolve_maxbits1d" => 14, "linsolve_maxbits2d" => 8)
for a in ARGS
    k, v = split(a, "="); opts[k] = parse(Int, v)
end

p = SIAParams()
dt = 100.0

# Sparse reference for d = 1, 2 from the face diffusivity arrays (same stencil as the MPOs)
function sparse_A(Dd::Vector, g, dt)
    n = npoints(g); h = g.h
    Id = sparse(1.0I, n, n)
    Df1 = spdiagm(0 => -ones(n), 1 => ones(n - 1)); Db1 = spdiagm(0 => ones(n), -1 => -ones(n - 1))
    if g.d == 1
        return I - dt * (Db1 / h[1]) * spdiagm(0 => vec(Dd[1])) * (Df1 / h[1])
    else
        # column-major vec: index = i + n*j  (x fastest)
        Dfx = kron(Id, Df1) / h[1]; Dbx = kron(Id, Db1) / h[1]
        Dfy = kron(Df1, Id) / h[2]; Dby = kron(Db1, Id) / h[2]
        return I - dt * (Dbx * spdiagm(0 => vec(Dd[1])) * Dfx + Dby * spdiagm(0 => vec(Dd[2])) * Dfy)
    end
end

for d in (1, 2)
    hb = Halfar(d = d, p = p)
    t0 = halfar_t0(hb)
    bits = d == 1 ? (10:2:opts["maxbits1d"]) : (6:opts["maxbits2d"])
    for nbits in bits
        g = QuanticsGrid(nbits, :interleaved, ntuple(_ -> (-1200e3, 1200e3), d)...)
        ops = SIAOperators(g)
        st = SolverSettings(linear_solver = :inverse)
        Hfun(t) = function_to_mps((x...) -> halfar_H(hb, t, sqrt(sum(abs2, x))), g; tolerance = 1e-8, maxbonddim = 128)
        H1, H2 = Hfun(t0), Hfun(t0 + dt)
        build(H) = begin
            D = face_diffusivities(H, nothing, ops, p, st)
            D, mpo_lincomb([1.0, -dt], [ops.I, diffusion_mpo(D, ops, st)]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
        end
        D1, A1 = build(H1)
        D2, A2 = build(H2)
        lmax = lambda_max_estimate(D1, dt, g)

        dense_ok = d * nbits <= 18
        ref1 = ref2 = nothing
        if dense_ok
            As1 = sparse_A([mps_to_array(Dk, g) for Dk in D1], g, dt); As2 = sparse_A([mps_to_array(Dk, g) for Dk in D2], g, dt)
            ref1 = As1 \ vec(mps_to_array(H1, g)); ref2 = As2 \ vec(mps_to_array(H2, g))
        end
        relerr(x, ref) = dense_ok ? maximum(abs, vec(mps_to_array(x, g)) - ref) / maximum(abs, ref) : NaN

        tr = InverseTracker()
        tc = @elapsed x1, i1 = solve_with_inverse!(tr, A1, H1, () -> lmax, ops, st)
        tw = @elapsed x2, i2 = solve_with_inverse!(tr, A2, H2, () -> lmax, ops, st)
        println(@sprintf("d=%d 2^%d^%d  chi(A) %3d  kappa~%.1e | inverse cold: %3d NS its %7.1f s chi(X) %3d res %.1e err %.1e | warm: %2d NS its %6.1f s chi(X) %3d err %.1e",
                         d, nbits, d, maxlinkdim(A1), lmax / 1.5, i1.ns_iters, tc, i1.X_maxdim, i1.residual, relerr(x1, ref1),
                         i2.ns_iters, tw, i2.X_maxdim, relerr(x2, ref2)))
        flush(stdout)
        if nbits <= (d == 1 ? opts["linsolve_maxbits1d"] : opts["linsolve_maxbits2d"])
            tl = @elapsed xl, _ = _linsolve(A2, H2, H2, st, true)
            println(@sprintf("      linsolve CG (warm guess): %7.1f s  err %.1e", tl, relerr(xl, ref2)))
            flush(stdout)
        end
    end
end
