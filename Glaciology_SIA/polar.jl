# SIA on a polar (r, theta) quantics grid: dimension 1 = r in [0, R] (cell centred, no cell at r = 0),
# dimension 2 = theta in [0, 2 pi) (periodic). For a near-radial ice sheet the margin is then close to
# a coordinate line, which is what keeps QTT ranks flat in 1D (a front at one point), whereas on a
# Cartesian grid a curved front makes them grow ~ 2^(n/2).
#
# Finite-volume form of the SIA with metric factors:
#   dH/dt = (1/r) d/dr (r D ds/dr) + (1/r^2) d/dtheta (D ds/dtheta) + a,
#   |grad s|^2 = (ds/dr)^2 + (ds/dtheta)^2 / r^2,   D = Gamma H^(n+2) |grad s|^(n-1)
# r-faces at r_(i+1/2) carry the flux r_f D_r ds/dr (zero at r = 0 because r_f = 0 there, so the
# zero ghost of the open shift is exact); theta-faces are periodic.

struct PolarOperators
    I::MPO
    Df::Vector{MPO}        # forward difference to the face (r: open, theta: periodic)
    Db::Vector{MPO}        # backward difference back to the cell
    Avg::Vector{MPO}       # cell -> face average
    Gc::Vector{MPO}        # central difference at cells
    rc::MPS                # cell radius
    rf::MPS                # r-face radius r + h_r/2
    inv_rc::MPS
    inv_rc2::MPS
    inv_rf::MPS
end

function PolarOperators(g::QuanticsGrid)

    g.d == 2 || error("polar grid is 2D")
    I = identity_mpo(g)
    pbc = (false, true)
    Sp = [shift_mpo(g, a, +1; pbc = pbc[a]) for a in 1:2]
    Sm = [shift_mpo(g, a, -1; pbc = pbc[a]) for a in 1:2]
    Df = [mpo_lincomb([1 / g.h[a], -1 / g.h[a]], [Sp[a], I]; cutoff = 1e-30) for a in 1:2]
    Db = [mpo_lincomb([1 / g.h[a], -1 / g.h[a]], [I, Sm[a]]; cutoff = 1e-30) for a in 1:2]
    Avg = [mpo_lincomb([0.5, 0.5], [I, Sp[a]]; cutoff = 1e-30) for a in 1:2]
    Gc = [mpo_lincomb([1 / (2g.h[a]), -1 / (2g.h[a])], [Sp[a], Sm[a]]; cutoff = 1e-30) for a in 1:2]
    hr = g.h[1]
    f(fun) = function_to_mps((r, th) -> fun(r), g; tolerance = 1e-15, maxbonddim = 64)

    return PolarOperators(I, Df, Db, Avg, Gc, f(r -> r), f(r -> r + hr / 2), f(r -> 1 / r), f(r -> 1 / r^2), f(r -> 1 / (r + hr / 2)))

end

# Face diffusivities [D on r-faces, D on theta-faces]
function polar_face_diffusivities(H::MPS, po::PolarOperators, p::SIAParams, st::SolverSettings)

    kw = prod_kw(st)
    G = Gamma(p)
    m = div(p.n_glen - 1, 2)
    # r-faces: ds/dr, and ds/dtheta / r averaged onto the face
    Hr = mps_apply(po.Avg[1], H; kw...)
    sr = mps_apply(po.Df[1], H; kw...)
    st_r = hadamard(po.inv_rf, mps_apply(po.Avg[1], mps_apply(po.Gc[2], H; kw...); kw...); kw...)
    g2r = lincomb([1.0, 1.0], [hadamard(sr, sr; kw...), hadamard(st_r, st_r; kw...)]; kw...)
    # theta-faces: ds/dtheta / r, and ds/dr averaged onto the face
    Ht = mps_apply(po.Avg[2], H; kw...)
    stt = hadamard(po.inv_rc, mps_apply(po.Df[2], H; kw...); kw...)
    sr_t = mps_apply(po.Avg[2], mps_apply(po.Gc[1], H; kw...); kw...)
    g2t = lincomb([1.0, 1.0], [hadamard(stt, stt; kw...), hadamard(sr_t, sr_t; kw...)]; kw...)

    Dr = G * mul_pow(hadamard_pow(g2r, m; kw...), Hr, p.n_glen + 2; kw...)
    Dt = G * mul_pow(hadamard_pow(g2t, m; kw...), Ht, p.n_glen + 2; kw...)

    return [Dr, Dt]

end

# L[D] H = (1/r) Db_r (r_f D_r Df_r H) + (1/r^2) Db_theta (D_theta Df_theta H) as one MPO
function polar_diffusion_mpo(D::Vector{MPS}, po::PolarOperators, st::SolverSettings)

    kw = (; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
    rD = hadamard(po.rf, D[1]; prod_kw(st)...)
    Lr = mpo_product(diag_mpo(po.inv_rc), mpo_product(po.Db[1], mpo_product(diag_mpo(rD), po.Df[1]; kw...); kw...); kw...)
    Lt = mpo_product(diag_mpo(po.inv_rc2), mpo_product(po.Db[2], mpo_product(diag_mpo(D[2]), po.Df[2]; kw...); kw...); kw...)

    return truncate!(exact_sum(Lr, Lt); kw...)

end
