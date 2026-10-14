# 2D SIA in front-fixing (Landau) coordinates with quantics MPS. Same discrete equations as
# dense_landau_2d.jl (see there for the metric): x = xi R(theta) e_r, unknowns h(xi, theta) (MPS, one
# 2D quantics grid, interleaved) and the margin R(theta) (Nt numbers).
#   r_h = xi R^2 h - xi R_old^2 h_old + dt [ Db1 FX + Db2 FT - xi R^2 a ]
#   FX = -D (ax h_xi + bx h_th) - x_f^2 R (Rdot+ S1+ h + Rdot- h)   on xi-faces (margin face masked)
#   FT = -D (at h_xi + bt h_th)                                     on theta-faces (periodic)
#   D  = Gamma H^5 (cxx h_xi^2 + cxt h_xi h_th + ctt h_th^2)
# All metric coefficients are fields of (xi, theta) built by TCI from R. The h-Jacobian is an exact
# MPO (coefficients times the averaging, normal and cross difference operators). The margin rows are
# coupled by quasi-Newton: per-theta Schur complements from the theta-decoupled operator J_xi, then
# dh = J^-1 (-r_h - b dR) with the full J (b = dr_h/dR at fixed R').
#   sbatch --time=02:00:00 slurm/run.sbatch landau_2d.jl case=eismint nbits=5 dt=1000 t_run=25000
using Printf, LinearAlgebra, HDF5
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl"); include("amen.jl")

cfg = Dict{String,Any}("case" => "eismint", "nbits" => 5, "dt" => 1000.0, "t_run" => 25000.0, "tol" => 1e-8, "max_iters" => 20,
                       "report" => 1, "R_init" => 300e3, "H_init" => 1000.0, "ell" => 0.2, "lin_cutoff" => 1e-14, "lin_maxdim" => 128,
                       "nsweeps" => 4, "noise_tol" => 1e-6, "tci_tol" => 1e-12, "check" => false, "cutoff" => 1e-16,
                       "prod_cutoff" => 1e-20, "op_cutoff" => 1e-18, "steady_tol" => 0.0, "fields_mode" => "sep",
                       "coupling" => "quasi", "gm_k" => 8, "gm_tol" => 1e-6, "fd" => "central", "rankdiag" => false, "acut" => 0.0,
                       "fd_eps" => 1e-6, "glob" => "hybrid", "verbose" => false, "jac_cutoff" => 0.0, "direct_maxdim" => 6000, "predict" => false,
                       "solver" => "dmrg", "amen_sweeps" => 8, "amen_tol" => 1e-6, "kick" => 4, "init" => "", "init_cutoff" => 1e-12, "iter_cutoff" => 0.0, "blk_cutoff" => 0.0)
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : cfg[k] isa Bool ? parse(Bool, v) : parse(typeof(cfg[k]), v)
end
p = SIAParams(); n = p.n_glen; G = Gamma(p); cv = G * (n / (2n + 1))^n
nb = cfg["nbits"]; N = 2^nb; Nt = N
g = QuanticsGrid(nb, :interleaved, (0.0, 1.0), (0.0, 2pi)); hx = 1 / N; ht = 2pi / N
xi = [(i - 0.5) * hx for i in 1:N]; th = [(j - 1) * ht for j in 1:Nt]
st = SolverSettings(cutoff = cfg["cutoff"], op_cutoff = cfg["op_cutoff"], prod_cutoff = cfg["prod_cutoff"], prod_maxdim = 256, op_maxdim = 512,
                    lin_cutoff = cfg["lin_cutoff"], lin_maxdim = cfg["lin_maxdim"], nsweeps = cfg["nsweeps"], lin_res_tol = 1.0, direct_maxdim = cfg["direct_maxdim"])
const TM = Dict("gm" => 0.0, "fields" => 0.0, "faces" => 0.0, "jac" => 0.0, "solve" => 0.0, "b" => 0.0)
kw = (; cutoff = st.cutoff, maxdim = st.maxdim); pk = prod_kw(st); okw = (; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
had(a, b) = hadamard(a, b; pk...)
lc(c, v) = lincomb(c, v; kw...)

# fields from (i, j) (1-based cell / face indices), theta node j at (j - 1) ht as in the dense code
function product_mps(vecs)
    psi = MPS(g.sites, fill("Up", nsites(g)))
    for k in 1:nsites(g)
        T = psi[k]; s = g.sites[k]; ls = [l for l in inds(T) if l != s]
        Tn = ITensor(inds(T)...)
        for v in 1:2
            Tn[s => v, (l => 1 for l in ls)...] = vecs[k][v]
        end
        psi[k] = Tn
    end
    return psi
end
ONE = product_mps(fill([1.0, 1.0], nsites(g)))
xisites = Set(g.pos[1])
e_xi(b) = product_mps([k in xisites ? (b == 0 ? [1.0, 0.0] : [0.0, 1.0]) : [1.0, 1.0] for k in 1:nsites(g)])
e_first = e_xi(0); e_last = e_xi(1)
function idxmps(f)
    vals = [f(i, j) for i in 1:4:N, j in 1:4:Nt]
    all(iszero, vals) && all(iszero, (f(N, j) for j in 1:Nt)) && return 0.0 * ONE
    _tci_to_mps(sigma -> (ij = bits_to_idx(g, sigma); f(ij[1] + 1, ij[2] + 1)), g; tolerance = cfg["tci_tol"], maxbonddim = 128)
end
Eval(m) = (E = MPSEvaluator(m); (i, j) -> E(idx_to_bits(g, (i - 1, j - 1))))
row(m, i) = (E = Eval(m); [E(i, j) for j in 1:Nt])

# exact MPS of a theta-only field v[j] (constant in xi): 1D MPS of v on the theta bits, embedded into the
# 2D site order with delta tensors (x [1, 1]) on the xi sites
g1 = QuanticsGrid(nb, :interleaved, (0.0, 2pi))
function theta_mps(v)
    all(iszero, v) && return 0.0 * ONE
    m1 = array_to_mps(v, g1; cutoff = 1e-15)
    L = nsites(g); out = Vector{ITensor}(undef, L)
    left = Index(1, "Link,emb0"); prev = left; k = 0
    for s in 1:L
        if s in g.pos[2]
            k += 1
            T = replaceind(m1[k], g1.sites[k], g.sites[s])
            T = k > 1 ? replaceind(T, linkind(m1, k - 1), prev) : T * onehot(prev => 1)
            if k < nb
                nl = sim(linkind(m1, k)); T = replaceind(T, linkind(m1, k), nl)
            else
                nl = Index(1, "Link,embR"); T = T * onehot(nl => 1)
            end
            out[s] = T; prev = nl
        else
            nl = sim(prev); out[s] = delta(prev, nl) * ITensor([1.0, 1.0], g.sites[s]); prev = nl
        end
    end
    out[1] = out[1] * onehot(left => 1); out[L] = out[L] * onehot(prev => 1)
    return MPS(out)
end
let v = [sin(3j) + 2 for j in 1:Nt], m = theta_mps(v), E = Eval(m)
    maximum(abs(E(i, j) - v[j]) for i in (1, N), j in (1, 2, Nt)) < 1e-10 || error("theta_mps check failed")
end
xi_mps(f) = idxmps((i, j) -> f(i))
xfv(i) = min(i, N - 1) * hx
XI = Dict(:xf => xi_mps(xfv), :ixf => xi_mps(i -> 1 / xfv(i)), :ixf2 => xi_mps(i -> 1 / xfv(i)^2), :xf2 => xi_mps(i -> xfv(i)^2),
          :x => xi_mps(i -> xi[i]), :ix => xi_mps(i -> 1 / xi[i]), :ix2 => xi_mps(i -> 1 / xi[i]^2))

# operators
Id = identity_mpo(g)
S1p = shift_mpo(g, 1, +1); S1m = shift_mpo(g, 1, -1)
S2p = shift_mpo(g, 2, +1; pbc = true); S2m = shift_mpo(g, 2, -1; pbc = true)
P1 = diag_mpo(lc([1.0, -1.0], [ONE, e_last]))
ml(c, v) = mpo_lincomb(c, v; cutoff = 1e-30)
mp(a, b) = mpo_product(a, b; cutoff = 1e-30, maxdim = typemax(Int))
D1 = mp(P1, ml([1 / hx, -1 / hx], [S1p, Id])); A1 = mp(P1, ml([0.5, 0.5], [Id, S1p])); Db1 = ml([1 / hx, -1 / hx], [Id, S1m])
D2 = ml([1 / ht, -1 / ht], [S2p, Id]); A2 = ml([0.5, 0.5], [Id, S2p]); Db2 = ml([1 / ht, -1 / ht], [Id, S2m])
Gc2 = ml([1 / (2ht), -1 / (2ht)], [S2p, S2m])
Gc1 = ml([1 / (2hx), -1 / (2hx), -1 / (2hx)], [S1p, S1m, diag_mpo(lc([1.0, 1.0], [e_first, e_last]))])   # ghosts h_0 = h_1, h_(N+1) = -h_N
A1Gc2 = mp(A1, Gc2); A2Gc1 = mp(A2, Gc1)
for (a, b) in ((e_first, (1, 1)), (e_last, (N, 3)))
    abs(Eval(a)(b...) - 1) < 1e-12 || error("product state check failed")
end

if cfg["case"] == "halfar"
    hb = Halfar(d = 2, p = p); t = halfar_t0(hb)
    amb(r, tt) = 0.0
    R = fill(halfar_margin(hb, t), Nt); h = idxmps((i, j) -> halfar_H(hb, t, xi[i] * R[j]))
else
    eb = EISMINTMovingMargin(p = p); ell = cfg["ell"]
    amb(r, tt) = cfg["case"] == "eismint" ? eismint_mb(eb, r) : eismint_mb(eb, r * sqrt((cos(tt) / (1 + ell))^2 + (sin(tt) / (1 - ell))^2))
    R = fill(cfg["R_init"], Nt); h = idxmps((i, j) -> cfg["H_init"] * sqrt(max(1 - xi[i]^2, 0.0))); t = 0.0
end
if cfg["init"] != ""
    # start from a saved dense state (dense_landau_2d.jl output, same grid): skips the start-up from the circle
    Hd, R, t = h5open(f -> (read(f, "h"), read(f, "R"), read(f, "t")), cfg["init"])
    size(Hd) == (N, Nt) || error("init grid $(size(Hd)) != $N x $Nt")
    h = truncate(idxmps((i, j) -> Hd[i, j]); cutoff = cfg["init_cutoff"])   # TCI in the same index convention as Eval
    Eh0 = Eval(h)
    ierr = maximum(abs(Eh0(i, j) - Hd[i, j]) for i in 1:max(1, N ÷ 64):N, j in 1:max(1, Nt ÷ 64):Nt) / maximum(Hd)
    println(@sprintf("init from %s: t = %.0f a, chi(h) %d, sampled max rel error %.1e", cfg["init"], t, maxlinkdim(h), ierr)); flush(stdout)
    ierr < 1e-3 || error("init MPS does not reproduce the array")
end
zero_mb = cfg["case"] == "halfar"
vfun(y) = y * abs(y)^((n + 1) / n); dvfun(y) = (2n + 1) / n * abs(y)^((n + 1) / n)

# coefficient fields for a given R, R_old, dt
fields(Rv, Rold, dt) = (t0 = time(); r = _fields(Rv, Rold, dt); TM["fields"] += time() - t0; r)
function _fields(Rv, Rold, dt)
    Rdot = (Rv .- Rold) ./ dt
    Rpc = [(Rv[mod1(j + 1, Nt)] - Rv[mod1(j - 1, Nt)]) / (2ht) for j in 1:Nt]
    Rf = [(Rv[j] + Rv[mod1(j + 1, Nt)]) / 2 for j in 1:Nt]; Rpf = [(Rv[mod1(j + 1, Nt)] - Rv[j]) / ht for j in 1:Nt]
    xf(i) = min(i, N - 1) * hx            # xi-face i + 1/2 (the masked margin face gets finite dummy values)
    F = Dict{Symbol,MPS}()
    if cfg["fields_mode"] == "sep"         # separable fields: (xi part) .* (theta part), exact
        T(v) = theta_mps(v); X(k, v) = had(XI[k], T(v))
        F[:cxx] = T((1 .+ (Rpc ./ Rv) .^ 2) ./ Rv .^ 2); F[:cxt] = X(:ixf, -2Rpc ./ Rv .^ 3); F[:ctt] = X(:ixf2, 1 ./ Rv .^ 2)
        F[:ax] = X(:xf, 1 .+ (Rpc ./ Rv) .^ 2); F[:bx] = T(-Rpc ./ Rv)
        F[:cp] = X(:xf2, Rv .* max.(Rdot, 0.0)); F[:cm] = X(:xf2, Rv .* min.(Rdot, 0.0))
        F[:txx] = T((1 .+ (Rpf ./ Rf) .^ 2) ./ Rf .^ 2); F[:txt] = X(:ix, -2Rpf ./ Rf .^ 3); F[:ttt] = X(:ix2, 1 ./ Rf .^ 2)
        F[:at] = T(-Rpf ./ Rf); F[:bt] = XI[:ix]; F[:m] = X(:x, Rv .^ 2)
        F[:ma] = zero_mb ? 0.0 * ONE : idxmps((i, j) -> xi[i] * Rv[j]^2 * amb(xi[i] * Rv[j], th[j]))
        return F, Rpc
    end
    F[:cxx] = idxmps((i, j) -> (1 + (Rpc[j] / Rv[j])^2) / Rv[j]^2)
    F[:cxt] = idxmps((i, j) -> -2Rpc[j] / (xf(i) * Rv[j]^3))
    F[:ctt] = idxmps((i, j) -> 1 / (xf(i) * Rv[j])^2)
    F[:ax] = idxmps((i, j) -> xf(i) * (1 + (Rpc[j] / Rv[j])^2))
    F[:bx] = idxmps((i, j) -> -Rpc[j] / Rv[j])
    F[:cp] = idxmps((i, j) -> xf(i)^2 * Rv[j] * max(Rdot[j], 0.0))
    F[:cm] = idxmps((i, j) -> xf(i)^2 * Rv[j] * min(Rdot[j], 0.0))
    F[:txx] = idxmps((i, j) -> (1 + (Rpf[j] / Rf[j])^2) / Rf[j]^2)
    F[:txt] = idxmps((i, j) -> -2Rpf[j] / (xi[i] * Rf[j]^3))
    F[:ttt] = idxmps((i, j) -> 1 / (xi[i] * Rf[j])^2)
    F[:at] = idxmps((i, j) -> -Rpf[j] / Rf[j])
    F[:bt] = idxmps((i, j) -> 1 / xi[i])
    F[:m] = idxmps((i, j) -> xi[i] * Rv[j]^2)
    F[:ma] = zero_mb ? 0.0 * ONE : idxmps((i, j) -> xi[i] * Rv[j]^2 * amb(xi[i] * Rv[j], th[j]))
    return F, Rpc
end

# truncation of a small field at a fixed fraction of the field it is added to (relative truncation
# keeps roundoff noise at full rank when the field itself is ~0, e.g. h_theta of an axisymmetric h)
atr(x, ref) = cfg["acut"] > 0 ? truncate(x; cutoff = cfg["acut"] * norm(ref)^2, use_absolute_cutoff = true) : x

# face quantities and fluxes
function faces(hc, F)
    o(op) = mps_apply(op, hc; pk...)
    Xgx = o(D1); Xgt = o(A1Gc2); Tgx = o(A2Gc1); Tgt = o(D2)
    rX = norm(Xgx) > norm(Xgt) ? Xgx : Xgt; rT = norm(Tgx) > norm(Tgt) ? Tgx : Tgt   # both gradients cut at the larger one's scale
    X = (H = o(A1), gx = atr(Xgx, rX), gt = atr(Xgt, rX)); T = (H = o(A2), gx = atr(Tgx, rT), gt = atr(Tgt, rT))
    g2(Q, cxx, cxt, ctt) = lc([1.0, 1.0, 1.0], [had(F[cxx], had(Q.gx, Q.gx)), had(F[cxt], had(Q.gx, Q.gt)), had(F[ctt], had(Q.gt, Q.gt))])
    w(Q, a, b) = lc([1.0, 1.0], [had(F[a], Q.gx), had(F[b], Q.gt)])
    gX = g2(X, :cxx, :cxt, :ctt); wX = w(X, :ax, :bx)
    gT = g2(T, :txx, :txt, :ttt); wT = atr(w(T, :at, :bt), wX)
    DX = G * mul_pow(gX, X.H, n + 2; pk...); DT = G * mul_pow(gT, T.H, n + 2; pk...)
    FX = lc([-1.0, -1.0, -1.0], [had(DX, wX), had(F[:cp], o(S1p)), had(F[:cm], o(P1))])
    FT = -1.0 * had(DT, wT)
    return X, T, gX, wX, gT, wT, FX, FT
end

function residual(hc, Rv, hold, Rold, dt)
    F, Rpc = fields(Rv, Rold, dt)
    mold = cfg["fields_mode"] == "sep" ? had(XI[:x], theta_mps(Rold .^ 2)) : idxmps((i, j) -> xi[i] * Rold[j]^2)
    t0 = time(); fc = faces(hc, F); TM["faces"] += time() - t0
    FX, FT = fc[7], fc[8]
    div = lc([1.0, 1.0], [mps_apply(Db1, FX; pk...), mps_apply(Db2, FT; pk...)])
    rh = lc([1.0, -1.0, dt, -dt], [had(F[:m], hc), had(mold, hold), div, F[:ma]])
    hN = row(hc, N); hN1 = row(hc, N - 1)
    rR = zeros(Nt); JRhN = zeros(Nt); JRhN1 = zeros(Nt); JRR = zeros(Nt)
    for j in 1:Nt
        fR(Rj) = (gx = sqrt(1 + (Rpc[j] / Rj)^2) / Rj; vxi = (-3vfun(hN[j]) + vfun(hN1[j]) / 3) / hx;
                  1e3 * hx * Rj * (Rj - Rold[j] + dt * cv * abs(vxi)^(n - 1) * vxi * gx^(n + 1) * Rj))
        rR[j] = fR(Rv[j])
        JRR[j] = imag(fR(complex(Rv[j], 1e-30 * Rv[j]))) / (1e-30 * Rv[j])
        gx = sqrt(1 + (Rpc[j] / Rv[j])^2) / Rv[j]; vxi = (-3vfun(hN[j]) + vfun(hN1[j]) / 3) / hx
        s = 1e3 * hx * Rv[j] * dt * cv * n * abs(vxi)^(n - 1) * gx^(n + 1) * Rv[j] / hx
        JRhN[j] = s * (-3dvfun(hN[j])); JRhN1[j] = s * (dvfun(hN1[j]) / 3)
    end
    return rh, rR, F, fc, (JRhN, JRhN1, JRR)
end
merit(rh, rR) = sqrt(norm(rh)^2 + norm(rR)^2)

# exact h-Jacobian (full, and theta-decoupled for the Schur complements)
function jacobians(hc, F, fc, dt; want_xi = true)
    X, T, gX, wX, gT, wT = fc
    H4X = mul_pow(G * ONE, X.H, n + 1; pk...); H5X = had(H4X, X.H)
    H4T = mul_pow(G * ONE, T.H, n + 1; pk...); H5T = had(H4T, T.H)
    # xi-faces
    a1 = -(n + 2) * had(H4X, had(gX, wX))
    dgx = lc([2.0, 1.0], [had(F[:cxx], X.gx), had(F[:cxt], X.gt)]); dgt = lc([1.0, 2.0], [had(F[:cxt], X.gx), had(F[:ctt], X.gt)])
    b1 = -1.0 * had(H5X, lc([1.0, 1.0], [had(dgx, wX), had(gX, F[:ax])]))
    c1 = -1.0 * had(H5X, lc([1.0, 1.0], [had(dgt, wX), had(gX, F[:bx])]))
    # theta-faces (gx = h_xi averaged, gt = h_th)
    a2 = -(n + 2) * had(H4T, had(gT, wT))
    dgxT = lc([2.0, 1.0], [had(F[:txx], T.gx), had(F[:txt], T.gt)]); dgtT = lc([1.0, 2.0], [had(F[:txt], T.gx), had(F[:ttt], T.gt)])
    b2 = -1.0 * had(H5T, lc([1.0, 1.0], [had(dgtT, wT), had(gT, F[:bt])]))     # with D2 (h_th)
    c2 = -1.0 * had(H5T, lc([1.0, 1.0], [had(dgxT, wT), had(gT, F[:at])]))     # with A2 Gc1 (h_xi)
    c1 = atr(c1, b1); a2 = atr(a2, a1); c2 = atr(c2, b2)
    if cfg["jac_cutoff"] > 0                       # Newton needs J only to ~1e-5: drop noise ranks of the coefficients
        jt(x) = truncate(x; cutoff = cfg["jac_cutoff"])
        a1, b1, c1, a2, b2, c2 = jt(a1), jt(b1), jt(c1), jt(a2), jt(b2), jt(c2)
    end
    tm(Db, c, op) = dt * mpo_product(Db, mpo_product(diag_mpo(c), op; okw...); okw...)
    Jxi_terms = MPO[tm(Db1, a1, A1), tm(Db1, b1, D1), tm(Db1, -1.0 * F[:cp], S1p), tm(Db1, -1.0 * F[:cm], P1)]
    Jt_terms = MPO[tm(Db1, c1, A1Gc2), tm(Db2, a2, A2), tm(Db2, b2, D2), tm(Db2, c2, A2Gc1)]
    Jfull = truncate!(exact_sum(diag_mpo(F[:m]), Jxi_terms..., Jt_terms...); okw...)
    if cfg["rankdiag"]
        @printf("  ranks: h %d | XH %d Xgx %d Xgt %d gX %d wX %d | TH %d Tgx %d Tgt %d gT %d wT %d | a1 %d b1 %d c1 %d a2 %d b2 %d c2 %d | terms %s | J %d\n",
                maxlinkdim(hc), maxlinkdim(X.H), maxlinkdim(X.gx), maxlinkdim(X.gt), maxlinkdim(gX), maxlinkdim(wX), maxlinkdim(T.H), maxlinkdim(T.gx),
                maxlinkdim(T.gt), maxlinkdim(gT), maxlinkdim(wT), maxlinkdim(a1), maxlinkdim(b1), maxlinkdim(c1), maxlinkdim(a2), maxlinkdim(b2),
                maxlinkdim(c2), string(maxlinkdim.([Jxi_terms; Jt_terms])), maxlinkdim(Jfull))
        for (k, v) in F; print(" ", k, ":", maxlinkdim(v)); end; println()
        for c in (1e-6, 1e-8, 1e-10, 1e-12)
            @printf("  J truncated at %.0e: chi %d;  a1 %d b1 %d c1 %d a2 %d b2 %d c2 %d\n", c, maxlinkdim(truncate(Jfull; cutoff = c)),
                    [maxlinkdim(truncate(q; cutoff = c)) for q in (a1, b1, c1, a2, b2, c2)]...)
        end
        flush(stdout)
    end
    want_xi || return Jfull, nothing
    Jxi = truncate!(exact_sum(diag_mpo(F[:m]), Jxi_terms...); okw...)
    return Jfull, Jxi
end

# ---------------------------------------------------------------------------------------------
# Bordered Newton system (coupling=bordered): h and R in ONE MPS on [flag; grid],
#   z = |1> (x) dh + |2> (x) (ones_xi (x) dR / cs),      K = [ J  B cs ; rs C  rs D cs ]
# B = d r_h / dR is theta-banded (column k of r_h depends on R_(k-1..k+1) through R, R', Rdot), so it is
# assembled exactly from 4 coloured forward differences of the residual (R_j perturbed for j = c mod 4):
#   B = sum_s diag(beta_s) S_theta^s,  beta_s(xi, k) = Delta_(colour(k+s))(xi, k).
# C copies the margin rows h(N), h(N-1) to every xi (so the R block is replicated in xi and its
# solution is xi-constant), D = d r_R / dR (also theta-banded). One linear solve per Newton iteration.
# ---------------------------------------------------------------------------------------------
flag = Index(2, "Flag")
function flag_mpo(a, b, A::MPO)
    M = zeros(2, 2); M[a, b] = 1.0
    l = Index(1, "Link,flag")
    MPO([itensor(M, prime(flag), flag) * onehot(l => 1); A[1] * onehot(l => 1); [A[k] for k in 2:length(A)]])
end
function flag_mps(a, x::MPS)
    v = zeros(2); v[a] = 1.0
    l = Index(1, "Link,flag")
    MPS([itensor(v, flag) * onehot(l => 1); x[1] * onehot(l => 1); [x[k] for k in 2:length(x)]])
end
unflag(z::MPS, a) = (v = zeros(2); v[a] = 1.0; MPS([z[1] * itensor(v, flag) * z[2]; [z[k] for k in 3:length(z)]]))
# (|ones><e_i| on the xi sites) (x) I_theta: copies row i of the input to every row
function bcast_mpo(i)
    bits = idx_to_bits(g, (i - 1, 0))              # site states (1 or 2) of cell (i, 1)
    L = nsites(g); links = [Index(1, "Link,bc$k") for k in 1:L-1]; Ts = ITensor[]
    for k in 1:L
        sk = g.sites[k]
        T = k in xisites ? (M = zeros(2, 2); M[:, bits[k]] .= 1.0; itensor(M, prime(sk), sk)) : itensor([1.0 0.0; 0.0 1.0], prime(sk), sk)
        k > 1 && (T *= onehot(links[k-1] => 1)); k < L && (T *= onehot(links[k] => 1))
        push!(Ts, T)
    end
    return MPO(Ts)
end
BC_N = bcast_mpo(N); BC_N1 = bcast_mpo(N - 1)
PN = diag_mpo(e_last); PNc = diag_mpo(lc([1.0, -1.0], [ONE, e_last]))
Ssh = Dict(-1 => S2m, 0 => Id, 1 => S2p)
let v = [sin(3j) + 2 for j in 1:Nt], m = theta_mps(v), hv = idxmps((i, j) -> i + 10j), E1 = Eval(mps_apply(Ssh[1], m; cutoff = 1e-30)),
    E2 = Eval(mps_apply(BC_N1, hv; cutoff = 1e-30)), E3 = Eval(unflag(flag_mps(2, hv), 2))
    abs(E1(3, 5) - v[6]) < 1e-10 && abs(E1(2, Nt) - v[1]) < 1e-10 || error("theta shift check failed")
    abs(E2(3, 5) - (N - 1 + 50)) < 1e-8 && abs(E2(N, 2) - (N - 1 + 20)) < 1e-8 || error("bcast check failed")
    abs(E3(4, 7) - (4 + 70)) < 1e-8 || error("flag check failed")
end
nrm_s(dh, dR, hc, Rc) = sqrt((norm(dh) / norm(hc))^2 + (norm(dR) / norm(Rc))^2)

function bordered_system(hc, Rc, rh, rR, F, JR, Jfull, h, R, dt)
    t0 = time()
    eps = cfg["fd_eps"] * sum(Rc) / Nt
    Dl = MPS[]; dRR = Vector{Float64}[]
    for c in 0:3
        pert = [mod(j - 1, 4) == c ? eps : 0.0 for j in 1:Nt]
        rp = residual(hc, Rc .+ pert, h, R, dt)
        d = lc([1 / eps, -1 / eps], [rp[1], rh]); cfg["jac_cutoff"] > 0 && (d = truncate(d; cutoff = cfg["jac_cutoff"]))
        push!(Dl, d); push!(dRR, (rp[2] .- rR) ./ eps)
    end
    TM["b"] += time() - t0
    JRhN, JRhN1, _ = JR
    # dR = cs * rho: rho (one row of N) must carry a norm comparable to dh (N rows), otherwise the
    # relative truncation of the linear solve is ~N times coarser on the margin unknowns
    cs = (sum(Rc) / Nt) / max(abs(row(hc, 1)[1]), 1.0) / sqrt(N)
    Bt = MPO[]; Dt = MPO[]
    for s in -1:1
        mask(c) = theta_mps([mod(k + s - 1, 4) == c ? 1.0 : 0.0 for k in 1:Nt])
        beta = lc(ones(4), [had(Dl[c+1], mask(c)) for c in 0:3]); cfg["jac_cutoff"] > 0 && (beta = truncate(beta; cutoff = cfg["jac_cutoff"]))
        delta_s = [dRR[mod(k + s - 1, 4)+1][k] for k in 1:Nt]
        push!(Bt, mpo_product(diag_mpo(beta), Ssh[s]; okw...)); push!(Dt, mpo_product(diag_mpo(theta_mps(delta_s)), Ssh[s]; okw...))
    end
    Bop = truncate!(exact_sum(Bt...); okw...); Dop = truncate!(exact_sum(Dt...); okw...)
    # block-wise truncation (each block relative to its own norm, so the small margin/R blocks survive)
    if cfg["blk_cutoff"] > 0
        Jfull = truncate(Jfull; cutoff = cfg["blk_cutoff"]); Bop = truncate(Bop; cutoff = cfg["blk_cutoff"])
    end
    Cop = truncate!(exact_sum(mpo_product(diag_mpo(theta_mps(JRhN)), BC_N; okw...), mpo_product(diag_mpo(theta_mps(JRhN1)), BC_N1; okw...)); okw...)
    # R lives on the margin row only: rho = e_N (x) dR / cs; the other rows of block 2 are identity (rho = 0)
    dmean = sum(abs, dRR[1] .+ dRR[2] .+ dRR[3] .+ dRR[4]) / Nt
    rs = (norm(F[:m]) / sqrt(N * Nt)) / (dmean * cs)            # R rows comparable to the mean diagonal of J
    B2 = mpo_product(Bop, BC_N; okw...)                         # reads rho on row N for every xi
    C2 = mpo_product(PN, Cop; okw...)
    D2 = truncate!(exact_sum(mpo_product(PN, Dop; okw...), dmean * PNc); okw...)
    K = truncate!(exact_sum(flag_mpo(1, 1, Jfull), flag_mpo(1, 2, cs * B2), flag_mpo(2, 1, rs * C2), flag_mpo(2, 2, (rs * cs) * D2)); okw...)
    return K, cs, rs, (maxlinkdim(Bop), maxlinkdim(Cop), maxlinkdim(Dop))
end
function bordered_solve(K, cs, rs, rh, rR, x0)
    rhs = lc([-1.0, -rs], [flag_mps(1, rh), flag_mps(2, had(e_last, theta_mps(rR)))])
    t0 = time()
    z, l = cfg["solver"] == "amen" ?
           amen_solve(K, rhs, x0 === nothing ? rhs : x0; nsweeps = cfg["amen_sweeps"], tol = cfg["amen_tol"], cutoff = cfg["lin_cutoff"],
                      maxdim = cfg["lin_maxdim"], kick = cfg["kick"], maxdirect = cfg["direct_maxdim"]) :
           _linsolve(K, rhs, x0 === nothing ? rhs : x0, st, false)
    TM["solve"] += time() - t0
    rho = unflag(z, 2)
    dR = cs .* row(rho, N)
    spread = maximum(abs, cs .* row(rho, N - 1)) / max(maximum(abs, dR), 1e-300)   # should be 0 off the margin row
    return unflag(z, 1), dR, z, l, spread
end

function be_step_bordered(h, R, dt; h0 = h, R0 = R)
    hc, Rc = h0, copy(R0); ch = Inf; it = 0; lin = NaN; chiJ = 0; zwarm = nothing
    rh, rR, F, fc, JR = residual(hc, Rc, h, R, dt)
    for m in 1:cfg["max_iters"]
        it = m
        t0 = time(); Jfull, _ = jacobians(hc, F, fc, dt; want_xi = false); TM["jac"] += time() - t0
        K, cs, rs, chs = bordered_system(hc, Rc, rh, rR, F, JR, Jfull, h, R, dt); chiJ = maxlinkdim(K)
        dh, dR, z, l, spread = bordered_solve(K, cs, rs, rh, rR, zwarm); lin = l[1]
        nd = nrm_s(dh, dR, hc, Rc); TM["gm"] += 1
        lam = 1.0; ok = false; local cand, dbar, nbar, zbar
        for _ in 0:6
            hn = lc([1.0, lam], [hc, dh]); Rn = Rc .+ lam .* dR
            # keep the Newton iterate at the rank of the solution: noise of inexact corrections otherwise accumulates
            cfg["iter_cutoff"] > 0 && (hn = truncate(hn; cutoff = cfg["iter_cutoff"]))
            cand = (hn, Rn, residual(hn, Rn, h, R, dt)...)
            if cfg["glob"] in ("nleqerr", "hybrid")
                # hybrid: Armijo on the residual norm (robust far from the solution), else the natural
                # monotonicity test (Deuflhard: simplified Newton correction with the old K), which is
                # not fooled by the truncation-noise floor of the residual near convergence
                okm = cfg["glob"] == "hybrid" && merit(cand[3], cand[4]) <= (1 - 1e-4lam) * merit(rh, rR)
                dbh, dbR, zbar, lb, _ = bordered_solve(K, cs, rs, cand[3], cand[4], z); TM["gm"] += 1
                nbar = nrm_s(dbh, dbR, hn, Rn); dbar = (dbh, dbR)
                ok = okm || nbar <= (1 - lam / 4) * nd
            else
                ok = merit(cand[3], cand[4]) <= (1 - 1e-4lam) * merit(rh, rR)
            end
            cfg["verbose"] && @printf("    newton %d: |d| %.2e  lam %.3f  %s  chi(h) %d chi(J) %d chi(K) %d (B %d C %d D %d)  lin res %.1e  rho spread %.1e  [%s]\n",
                                      m, nd, lam, cfg["glob"] != "merit" ? @sprintf("|dbar| %.2e", nbar) : @sprintf("merit %.2e", merit(cand[3], cand[4])),
                                      maxlinkdim(hc), maxlinkdim(Jfull), chiJ, chs..., lin, spread, join([@sprintf("%s %.0f", k, v) for (k, v) in TM], " "))
            flush(stdout)
            ok && break
            lam /= 2
        end
        ch = nd
        if !ok
            nd < cfg["noise_tol"] && return lc([1.0, 1.0], [hc, dh]), Rc .+ dR, true, it, ch, lin, chiJ
            return hc, Rc, false, it, ch, lin, chiJ
        end
        hc, Rc, rh, rR, F, fc, JR = cand
        if cfg["glob"] in ("nleqerr", "hybrid")
            zwarm = zbar
            if nbar < cfg["tol"]                     # converged: add the (tiny) simplified correction
                return lc([1.0, 1.0], [hc, dbar[1]]), Rc .+ dbar[2], true, it, nbar, lin, chiJ
            end
        end
        ch < cfg["tol"] && break
    end
    return hc, Rc, ch < cfg["tol"], it, ch, lin, chiJ
end

function be_step(h, R, dt; kws...)
    cfg["coupling"] == "bordered" && return be_step_bordered(h, R, dt; kws...)
    hc, Rc = h, copy(R); ch = Inf; it = 0; lin = NaN; chiJ = 0
    rh, rR, F, fc, JR = residual(hc, Rc, h, R, dt)
    for m in 1:cfg["max_iters"]
        it = m
        t0 = time(); Jfull, Jxi = jacobians(hc, F, fc, dt); TM["jac"] += time() - t0; chiJ = maxlinkdim(Jfull)
        # b = d r_h / dR at fixed R' (uniform shift of R), central difference
        epsR = 1e-6 * sum(Rc) / Nt
        t0 = time(); rp = residual(hc, Rc .+ epsR, h, R, dt)[1]
        rm = cfg["fd"] == "forward" ? rh : residual(hc, Rc .- epsR, h, R, dt)[1]; TM["b"] += time() - t0
        cb = cfg["fd"] == "forward" ? 1 / epsR : 1 / (2epsR)
        b = lc([cb, -cb], [rp, rm])
        JRhN, JRhN1, JRR = JR
        Cv(y) = (yN = row(y, N); yN1 = row(y, N - 1); JRhN .* yN .+ JRhN1 .* yN1)
        t0 = time()
        w, l2 = _linsolve(Jxi, b, b, st, false)
        Mdiag = JRR .- Cv(w)                         # per-theta Schur complement of the theta-decoupled operator
        if cfg["coupling"] == "gmres"
            # exact Newton on (h, R): GMRES on the Schur complement S = dR_R - C J^-1 B (B = dr_h/dR incl. R'),
            # preconditioned by Mdiag; B v and dR_R v by central differences of the residual along v
            x1, l1 = _linsolve(Jfull, -1.0 * rh, -1.0 * rh, st, false); TM["solve"] += time() - t0
            Ys = MPS[]; ls = Any[l1, l2]
            function Smul(v)
                e = 1e-6 * norm(Rc) / norm(v)
                t1 = time(); rp = residual(hc, Rc .+ e .* v, h, R, dt)
                rm = cfg["fd"] == "forward" ? (rh, rR) : residual(hc, Rc .- e .* v, h, R, dt); TM["b"] += time() - t1
                c = cfg["fd"] == "forward" ? 1 / e : 1 / (2e)
                Bv = lc([c, -c], [rp[1], rm[1]]); DRv = (rp[2] .- rm[2]) .* c
                t1 = time(); y, ly = _linsolve(Jfull, Bv, Bv, st, false); TM["solve"] += time() - t1
                push!(Ys, y); push!(ls, ly)
                return DRv .- Cv(y)
            end
            rhs0 = -rR .- Cv(x1)                     # S dR = -r_R - C J^-1 (-r_h)
            # right-preconditioned GMRES, z = M^-1 u, basis vectors v_k = M^-1 q_k
            beta = norm(rhs0); Q = [rhs0 ./ beta]; Hh = zeros(cfg["gm_k"] + 1, cfg["gm_k"]); Vs = Vector{Float64}[]; kk = 0; ym = Float64[]
            for k in 1:cfg["gm_k"]
                kk = k; v = Q[k] ./ Mdiag; push!(Vs, v)
                u = Smul(v)
                for i in 1:k; Hh[i, k] = dot(Q[i], u); u = u .- Hh[i, k] .* Q[i]; end
                Hh[k+1, k] = norm(u); push!(Q, u ./ max(Hh[k+1, k], 1e-300))
                e1 = zeros(k + 1); e1[1] = beta
                ym = Hh[1:k+1, 1:k] \ e1
                norm(Hh[1:k+1, 1:k] * ym .- e1) < cfg["gm_tol"] * beta && break
            end
            TM["gm"] += kk
            dR = sum(ym[i] .* Vs[i] for i in 1:kk)
            # dh = J^-1(-r_h) - J^-1 B dR, with J^-1 B v_i = Ys[i] (B linear in v up to FD error)
            dh = lc([1.0; -ym[1:kk]], vcat([x1], Ys[1:kk]))
            lin = maximum(l[1] for l in ls)
        else
            x1, l1 = _linsolve(Jxi, -1.0 * rh, -1.0 * rh, st, false)
            TM["solve"] += time() - t0
            x1N, x1N1 = row(x1, N), row(x1, N - 1)
            dR = (-rR .- JRhN .* x1N .- JRhN1 .* x1N1) ./ Mdiag
            dRf = theta_mps(dR)
            rhs = lc([-1.0, -1.0], [rh, had(b, dRf)])
            t0 = time(); dh, l3 = _linsolve(Jfull, rhs, x1, st, false); TM["solve"] += time() - t0
            lin = max(l1[1], l2[1], l3[1])
        end
        m0 = merit(rh, rR); lam = 1.0; ok = false
        local cand
        for _ in 0:8
            hn = lc([1.0, lam], [hc, dh]); Rn = Rc .+ lam .* dR
            # keep the Newton iterate at the rank of the solution: noise of inexact corrections otherwise accumulates
            cfg["iter_cutoff"] > 0 && (hn = truncate(hn; cutoff = cfg["iter_cutoff"]))
            cand = (hn, Rn, residual(hn, Rn, h, R, dt)...)
            ok = merit(cand[3], cand[4]) <= (1 - 1e-4lam) * m0
            ok && break
            lam /= 2
        end
        ch = norm(dh) / norm(hc) + norm(dR) / norm(Rc)
        if !ok
            ch < cfg["noise_tol"] && return lc([1.0, 1.0], [hc, dh]), Rc .+ dR, true, it, ch, lin, chiJ
            return hc, Rc, false, it, ch, lin, chiJ
        end
        hc, Rc, rh, rR, F, fc, JR = cand
        ch < cfg["tol"] && break
    end
    return hc, Rc, ch < cfg["tol"], it, ch, lin, chiJ
end

println(@sprintf("Landau 2D MPS %s: %d x %d (xi x theta), dt = %g a, lin_cutoff %g", cfg["case"], N, Nt, cfg["dt"], cfg["lin_cutoff"]))
if cfg["check"]
    # J v against central differences of the residual (R fixed)
    dt = cfg["dt"]; F, _ = fields(R, R, dt); fc = faces(h, F); Jfull, _ = jacobians(h, F, fc, dt)
    v = idxmps((i, j) -> 10 * sin(pi * xi[i])^2 * (1 + 0.3cos(th[j])))
    for e in (1e-1, 1e-2, 1e-3)
        rp = residual(lc([1.0, e], [h, v]), R, h, R, dt)[1]; rm = residual(lc([1.0, -e], [h, v]), R, h, R, dt)[1]
        fd = lc([1 / (2e), -1 / (2e)], [rp, rm]); Jv = mps_apply(Jfull, v; cutoff = 1e-30, maxdim = typemax(Int))
        @printf("eps %.0e: |Jv - FD| / |Jv| = %.2e\n", e, norm(lc([1.0, -1.0], [Jv, fd])) / norm(Jv))
    end
end
t_end = t + cfg["t_run"]; dt = cfg["dt"]; nst = 0; nit = 0; nrej = 0
hprev = nothing; Rprev = nothing; dtprev = 0.0
wall = @elapsed while t < t_end - 1e-9
    global h, R, t, dt, nst, nit, nrej, hprev, Rprev, dtprev
    dtt = min(dt, t_end - t)
    # predictor: linear extrapolation of the last two time levels as the Newton initial guess
    pk0 = cfg["predict"] && hprev !== nothing ? (; h0 = lc([1 + dtt / dtprev, -dtt / dtprev], [h, hprev]), R0 = R .+ (dtt / dtprev) .* (R .- Rprev)) : (;)
    ts = @elapsed (hn, Rn, ok, it, ch, lin, chiJ) = be_step(h, R, dtt; pk0...)
    if !ok
        nrej += 1; dt /= 2
        println(@sprintf("  rejected (it %d, change %.1e, lin res %.1e), dt -> %g", it, ch, lin, dt)); dt < 1e-3 && error("dt collapsed"); continue
    end
    rate = norm(Rn - R) / norm(R) / dtt
    hprev, Rprev, dtprev = h, R, dtt
    h, R, t = hn, Rn, t + dtt; nst += 1; nit += it
    it <= 6 && (dt = min(cfg["dt"], 1.5dt))
    nst % cfg["report"] == 0 && println(@sprintf("t = %9.1f a  dt %7.1f  R in [%.3f, %.3f] km  h(pole) %.2f m  newton %d  lin res %.1e  chi(h) %d (x%.1f)  chi(J) %d  %.1f s  [fields %.1f faces %.1f jac %.1f b %.1f solve %.1f gmres %d]",
                                                 t, dtt, minimum(R) / 1e3, maximum(R) / 1e3, Eval(h)(1, 1), it, lin, maxlinkdim(h), compression(h, g), chiJ, ts,
                                                 TM["fields"], TM["faces"], TM["jac"], TM["b"], TM["solve"], TM["gm"]))
    for k in keys(TM); TM[k] = 0.0; end
    flush(stdout)
    cfg["steady_tol"] > 0 && rate < cfg["steady_tol"] && break
end
@printf("%d steps (%d rejected), mean Newton %.1f, %.1f s; chi(h) %d, %d numbers for %d cells (x%.1f)\n", nst, nrej, nit / max(nst, 1), wall,
        maxlinkdim(h), nparams(h), N * Nt, compression(h, g))
if cfg["case"] == "halfar"
    Re = halfar_margin(hb, t)
    @printf("t = %.0f a: margin %.4f .. %.4f km (exact %.4f), pole %.3f m (exact %.3f)\n", t, minimum(R) / 1e3, maximum(R) / 1e3, Re / 1e3,
            Eval(h)(1, 1), halfar_H(hb, t, xi[1] * Re))
else
    @printf("t = %.0f a: margin %.3f .. %.3f km, h(pole) %.2f m\n", t, minimum(R) / 1e3, maximum(R) / 1e3, Eval(h)(1, 1))
end
