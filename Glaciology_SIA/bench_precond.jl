# Local solver of linsolve: KrylovKit GMRES (old default) vs dense direct LU of the projected
# two-site problem, on the two 1D systems of an implicit SIA step (Halfar, dt = 10 a):
#   Picard   (I - dt L[D(H0)]) x = H0                      (SPD)
#   Newton   J(H1) dH = -R(H1),  R = H1 - dt F(H1) - H0     (non-symmetric, the one that failed)
# against sparse direct solves of the same discrete systems.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_precond.jl bits=10,12
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "10,12", "dt" => "10", "sweeps" => "2,4")
for a in ARGS
    k, v = split(a, "="); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
dt = parse(Float64, opts["dt"])
sweeps = parse.(Int, split(opts["sweeps"], ","))

p = SIAParams()
hb = Halfar(d = 1, p = p)
t0 = halfar_t0(hb)
G = Gamma(p)

for nbits in bits
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3))
    ops = SIAOperators(g)
    st0 = SolverSettings()
    H0 = function_to_mps(x -> halfar_H(hb, t0, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    H1 = function_to_mps(x -> halfar_H(hb, t0 + dt, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)

    # MPS systems
    F0 = face_fields(H0, nothing, ops, p, st0)
    A = mpo_lincomb([1.0, -dt], [ops.I, diffusion_mpo([f.D for f in F0], ops, st0)]; cutoff = st0.op_cutoff, maxdim = st0.op_maxdim)
    F1 = face_fields(H1, nothing, ops, p, st0)
    R = lincomb([1.0, -dt, -1.0], [H1, flux_divergence(F1, ops, st0), H0]; cutoff = st0.cutoff, maxdim = st0.maxdim)
    J = mpo_lincomb([1.0, -dt], [ops.I, jacobian_mpo(F1, ops, p, st0)]; cutoff = st0.op_cutoff, maxdim = st0.op_maxdim)
    mR = -1.0 * R

    # sparse references
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    h0 = mps_to_array(H0, g); h1 = mps_to_array(H1, g)
    Dface(H) = (He = Avg * H; s = Df * H; G .* He .^ 5 .* s .^ 2)
    As = I - dt * Db * spdiagm(0 => Dface(h0)) * Df
    He = Avg * h1; gg = Df * h1
    Js = I - dt * Db * (spdiagm(0 => 3G .* He .^ 5 .* gg .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* gg .^ 3) * Avg)
    Rs = h1 - dt * Db * (Dface(h1) .* (Df * h1)) - h0
    xA = As \ h0; xJ = Js \ (-Rs)
    xJm = Js \ mps_to_array(mR, g)     # same sparse J, but the MPS right-hand side
    dR = mps_to_array(R, g) - Rs; imax = argmax(abs.(dR))
    println(@sprintf("   R(MPS) - R(sparse): max %.2e at x = %.1f km (H there %.1f m), rel %.1e; sparse dH from MPS rhs vs sparse rhs: rel diff %.1e",
                     maximum(abs, dR), coord(g, 1, imax - 1) / 1e3, h1[imax], norm(dR) / norm(Rs), norm(xJm - xJ) / norm(xJ)))
    kappa = 1 + 4dt * maximum(Dface(h0)) / h^2
    println(@sprintf("\n2^%d points: chi(H) %d, chi(A) %d, chi(J) %d, kappa ~ %.1e, |R|/|H| %.1e (MPS) %.1e (sparse)",
                     nbits, maxlinkdim(H0), maxlinkdim(A), maxlinkdim(J), kappa, norm(R) / norm(H1), norm(Rs) / norm(h1)))
    flush(stdout)

    for (sysname, Op, rhs, ref, spd, x0) in (("picard", A, H0, xA, true, H0), ("newton", J, mR, xJm, false, mR))
        for (lname, ls) in ((:krylov, :krylov), (:direct, :direct)), ns in sweeps
            st = SolverSettings(local_solver = ls)
            local x
            t = @elapsed ok = try
                x = _linsolve_once(Op, rhs, x0, st, spd, ns); true
            catch e
                println("   ", sysname, " ", lname, " nsweeps ", ns, ": ", sprint(showerror, e)[1:min(end, 120)]); false
            end
            ok || continue
            res = residual_norm(Op, x, rhs)
            err = norm(mps_to_array(x, g) - ref) / norm(ref)
            println(@sprintf("   %-7s %-7s nsweeps %d: %7.2f s  chi(x) %3d  residual %.1e  rel err vs sparse %.1e",
                             sysname, lname, ns, t, maxlinkdim(x), res, err))
            flush(stdout)
        end
    end
end
