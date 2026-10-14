# One complementarity Newton iteration near the EISMINT steady state, MPS vs dense evaluation of the
# same quantities on the same H (read from a run's output): Phi = m R + (1-m) H, the mask
# m = 1[H >= R], the Newton direction of (I - dt diag(m) dF/dH) dH = -Phi, and the merit ||Phi||
# along the MPS and the dense directions (both evaluated densely and in MPS).
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_ncp.jl file=output/<eismint run>.h5 dt=500
using LinearAlgebra, SparseArrays, Printf, HDF5
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("file" => "", "dt" => "500", "ordering" => "interleaved", "nbits" => "0", "lin_maxdim" => "32")
for a in ARGS
    k, v = split(a, "=", limit = 2); opts[k] = v
end
dt = parse(Float64, opts["dt"])
Hm, nbits, ord = h5open(opts["file"], "r") do f
    read(f, "H_mps", MPS), read(attributes(f)["nbits"]), Symbol(read(attributes(f)["ordering"]))
end
p = SIAParams(); eb = EISMINTMovingMargin(p = p); L = eb.L
gc = QuanticsGrid(nbits, ord, (-L, L))
nb = parse(Int, opts["nbits"])
if nb > nbits              # exact prolongation of the stored state to a finer grid (restart)
    g = QuanticsGrid(nb, ord, (-L, L)); Hm = prolong_mps(Hm, gc, g); nbits = nb
else
    g = gc
end
ops = SIAOperators(g)
st = SolverSettings(cutoff = 1e-20, op_cutoff = 1e-22, prod_cutoff = 1e-24, prod_maxdim = 256, lin_maxdim = parse(Int, opts["lin_maxdim"]), lin_cutoff = 1e-12,
                    lin_res_tol = 0.1, complementarity = true)
kw = (; cutoff = st.cutoff, maxdim = st.maxdim)
H = replace_siteinds(Hm, g.sites)
a_mb = function_to_mps(x -> eismint_mb(eb, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
rhs0 = lincomb([1.0, dt], [H, a_mb]; kw...)                 # backward Euler from H

# --- MPS (as in implicit_step)
residual(Hc) = lincomb([1.0, -dt, -1.0], [Hc, flux_divergence(face_fields(Hc, nothing, ops, p, st), ops, st), rhs0]; kw...)
mask_of(Hc, Rc) = indicator_mps((h, r) -> h - r, (Hc, Rc), g; tolerance = st.ncp_tol, maxbonddim = st.maxdim)
phi(Hc, Rc, mk) = lincomb([1.0, 1.0, -1.0], [hadamard(mk, Rc; prod_kw(st)...), Hc, hadamard(mk, Hc; prod_kw(st)...)]; kw...)
merit_mps(Hc) = (R = residual(Hc); norm(phi(Hc, R, mask_of(Hc, R))))
R = residual(H); mask = mask_of(H, R); Phi = phi(H, R, mask)
F = face_fields(H, nothing, ops, p, st)
Jac = mpo_product(diag_mpo(mask), jacobian_mpo(F, ops, p, st); cutoff = st.op_cutoff, maxdim = st.op_maxdim)
J = mpo_lincomb([1.0, -dt], [ops.I, Jac]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
dH, linfo = _linsolve(J, -1.0 * Phi, -1.0 * Phi, st, false)

# --- dense, same scheme on the same H values
N = 2^nbits; h = g.h[1]
Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2; G = Gamma(p)
Fd(Hv) = (He = Avg * Hv; s = Df * Hv; Db * (G .* He .^ 5 .* s .^ 2 .* s))
Jd(Hv) = (He = Avg * Hv; s = Df * Hv; Db * (spdiagm(0 => 3G .* He .^ 5 .* s .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* s .^ 3) * Avg))
hv = mps_to_array(H, g); av = mps_to_array(a_mb, g); r0 = hv + dt * av
Rd(Hv) = Hv - dt * Fd(Hv) - r0
Phid(Hv) = (Rv = Rd(Hv); m = Hv .>= Rv; ifelse.(m, Rv, Hv))
Rv = Rd(hv); md = hv .>= Rv; Phv = Phid(hv)
Jdm = I - dt * spdiagm(0 => Float64.(md)) * Jd(hv)
dHd = Jdm \ (-Phv)
x = coords(g, 1)

dv = mps_to_array(dH, g)
println(@sprintf("2^%d, dt = %g a: |Phi| dense %.3e (rel to |rhs| %.2e), MPS %.3e; |Phi_mps - Phi_dense|/|Phi_dense| %.2e",
                 nbits, dt, norm(Phv), norm(Phv) / norm(r0), norm(Phi), norm(mps_to_array(Phi, g) - Phv) / norm(Phv)))
println(@sprintf("mask cells that differ: %d (active cells dense %d); |R_mps - R_dense| max %.2e at x = %.1f km",
                 count(round.(mps_to_array(mask, g)) .!= md), count(.!md), maximum(abs, mps_to_array(R, g) - Rv),
                 x[argmax(abs.(mps_to_array(R, g) - Rv))] / 1e3))
i = argmax(abs.(Phv)); println(@sprintf("largest |Phi| dense: %.3e at x = %.2f km (H %.3f m, R %.3e, mask %d)", Phv[i], x[i] / 1e3, hv[i], Rv[i], md[i]))
println(@sprintf("chi(J) %d, lin res %.1e (%d), chi(dH) %d; |dH_mps - dH_dense|/|dH_dense| %.2e, |dH_dense|/|H| %.2e",
                 maxlinkdim(J), linfo[1], linfo[2], maxlinkdim(dH), norm(dv - dHd) / norm(dHd), norm(dHd) / norm(hv)))
j = argmax(abs.(dv - dHd)); println(@sprintf("largest direction difference at x = %.2f km: MPS %.3e dense %.3e", x[j] / 1e3, dv[j], dHd[j]))
# exact solve with the MPS operator (as a dense matrix through the MPO acting on unit vectors would be costly):
# the residual of the dense direction in the MPS operator
dHd_m = array_to_mps(dHd, g; cutoff = 1e-24)
println(@sprintf("dense direction as MPS: chi %d; its residual in the MPS system %.2e", maxlinkdim(dHd_m), residual_norm(J, dHd_m, -1.0 * Phi)))
for lam in (1.0, 0.5, 0.25, 0.125)
    m1 = norm(Phid(hv + lam * dv)) / norm(Phv); m2 = norm(Phid(hv + lam * dHd)) / norm(Phv)
    m3 = merit_mps(lincomb([1.0, lam], [H, dH]; kw...)) / norm(Phi); m4 = merit_mps(lincomb([1.0, lam], [H, dHd_m]; kw...)) / norm(Phi)
    println(@sprintf("lambda %5.3f: merit ratio  dense eval: MPS dir %.3e  dense dir %.3e | MPS eval: MPS dir %.3e  dense dir %.3e", lam, m1, m2, m3, m4))
end
