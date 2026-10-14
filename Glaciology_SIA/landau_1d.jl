# 1D SIA in front-fixing (Landau) coordinates with quantics MPS (see dense_landau_1d.jl for the
# formulation). x = xi R(t), xi in [0, 1]; unknowns h(xi) (MPS) and the margin position R (scalar).
#   r_h = R h - R_old h_old + dt [ -R^-n FD(h) - Rdot Db P (xi_f .* U h) - R a(xi R) ]
#   r_R = R - R_old + dt c_v |v_x|^(n-1) v_x,   v = h |h|^((n+1)/n) at xi = 1 (one-sided)
# FD = div_xi(D_xi grad_xi h) on the unit grid (the existing face_fields / flux_divergence /
# jacobian_mpo, since grad_x = grad_xi / R gives q = R^-n q_xi). P zeroes the margin face xi = 1 (no
# flux leaves the ice: the margin moves with it), the symmetric face xi = 0 is zero through the open
# shift. U = I or S+ (upwind for the transport -xi Rdot). Newton on (h, R) by block elimination: two
# MPS linear solves with the same J_hh and a scalar Schur complement.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch landau_1d.jl case=halfar nbits=12 dt=100
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

cfg = Dict{String,Any}("case" => "halfar", "nbits" => 10, "dt" => 100.0, "t_run" => 20000.0, "tol" => 1e-8, "max_iters" => 15,
                       "report" => 1, "R_init" => 300e3, "H_init" => 1000.0, "lin_cutoff" => 1e-16, "lin_maxdim" => 128,
                       "nsweeps" => 4, "lin_tol" => 0.0, "noise_tol" => 1e-6, "tci_tol" => 1e-12, "check" => true)
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : cfg[k] isa Bool ? parse(Bool, v) : parse(typeof(cfg[k]), v)
end
p = SIAParams(); n = p.n_glen; G = Gamma(p); cv = G * (n / (2n + 1))^n
nb = cfg["nbits"]; N = 2^nb
g = QuanticsGrid(nb, :interleaved, (0.0, 1.0)); hx = g.h[1]
st = SolverSettings(cutoff = 1e-20, op_cutoff = 1e-22, prod_cutoff = 1e-24, prod_maxdim = 256, lin_cutoff = cfg["lin_cutoff"],
                    lin_maxdim = cfg["lin_maxdim"], nsweeps = cfg["nsweeps"], lin_tol = cfg["lin_tol"], lin_res_tol = 1.0)
kw = (; cutoff = st.cutoff, maxdim = st.maxdim); okw = (; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
ONE = function_to_mps(x -> 1.0, g; tolerance = 1e-14, maxbonddim = 4)
# TCI of f; an identically zero f (TCI cannot normalise it, e.g. da/dr where a is flat) -> 0 MPS
tomps(f; tol = cfg["tci_tol"]) = all(f((i - 0.5) / 64) == 0 for i in 1:64) && f(1 - hx / 2) == 0 && f(hx / 2) == 0 ? 0.0 * ONE :
                                 function_to_mps(f, g; tolerance = tol, maxbonddim = 64)

# operators: margin face masked
ops0 = SIAOperators(g)
elast = MPS(g.sites, fill("Dn", nb))
E = MPSEvaluator(elast); bits(i) = idx_to_bits(g, (i,))
(E(bits(N - 1)) == 1 && E(bits(N - 2)) == 0) || error("last-cell state is not |1...1>")
mask = lincomb([1.0, -1.0], [tomps(x -> 1.0), elast]; cutoff = 1e-30)
P = diag_mpo(mask)
mp(A) = mpo_product(P, A; cutoff = 1e-30, maxdim = typemax(Int))
PDf = mp(ops0.Df[1])
ops = SIAOperators(ops0.I, [PDf], ops0.Db, [mp(ops0.Avg[1])], reshape([PDf], 1, 1), ops0.Gcell)
PSp = mp(shift_mpo(g, 1, +1))
xif = tomps(x -> x + hx / 2)
DbXP = [mpo_product(ops.Db[1], mpo_product(diag_mpo(xif), U; okw...); okw...) for U in (P, PSp)]   # Db diag(xi_f) P U

if cfg["case"] == "halfar"
    hb = Halfar(d = 1, p = p); t = halfar_t0(hb)
    amb(r) = 0.0; damb(r) = 0.0
    R = halfar_margin(hb, t); h = tomps(x -> halfar_H(hb, t, x * R))
else
    eb = EISMINTMovingMargin(p = p)
    amb(r) = eismint_mb(eb, r); damb(r) = eb.S * (eb.R_el - r) < eb.Mmax ? -eb.S : 0.0
    R = cfg["R_init"]; t = 0.0; h = tomps(x -> cfg["H_init"] * sqrt(max(1 - x^2, 0.0)))
end
zero_mb = cfg["case"] == "halfar"

vfun(y) = y * abs(y)^((n + 1) / n); dvfun(y) = (2n + 1) / n * abs(y)^((n + 1) / n)
phi(y) = abs(y)^(n - 1) * y
front_vals(hc) = (Ec = MPSEvaluator(hc); (Ec(bits(N - 1)), Ec(bits(N - 2))))

function residuals(hc, Rc, hold, Rold, dt)
    Rdot = (Rc - Rold) / dt
    F = face_fields(hc, nothing, ops, p, st)
    FD = flux_divergence(F, ops, st)
    adv = mps_apply(DbXP[Rdot < 0 ? 1 : 2], hc; prod_kw(st)...)
    terms = [Rc, -Rold, -dt * Rc^(-n), -dt * Rdot]; parts = [hc, hold, FD, adv]
    if !zero_mb
        aR = tomps(x -> amb(x * Rc)); push!(terms, -dt * Rc); push!(parts, aR)
    end
    rh = lincomb(terms, parts; kw...)
    hN, hN1 = front_vals(hc)
    vx = (-3vfun(hN) + vfun(hN1) / 3) / (hx * Rc)
    rR = 1e3 * (Rc - Rold + dt * cv * phi(vx))
    return rh, rR, F, FD, adv, (hN, hN1, vx), Rdot
end
merit(rh, rR) = sqrt(norm(rh)^2 + rR^2)

function be_step(h, R, dt)
    hc, Rc = h, R; ch = Inf; it = 0; lin = (NaN, 0); chiJ = 0
    rh, rR, F, FD, adv, fv, Rdot = residuals(hc, Rc, h, R, dt)
    for m in 1:cfg["max_iters"]
        it = m
        hN, hN1, vx = fv
        JF = jacobian_mpo(F, ops, p, st)
        Jhh = mpo_lincomb([Rc, -dt * Rc^(-n), -dt * Rdot], [ops.I, JF, DbXP[Rdot < 0 ? 1 : 2]]; okw...)
        chiJ = maxlinkdim(Jhh)
        terms = [1.0, dt * n * Rc^(-n - 1), -1.0]; parts = [hc, FD, adv]
        if !zero_mb
            push!(terms, -dt, -dt * Rc); push!(parts, tomps(x -> amb(x * Rc)), tomps(x -> x * damb(x * Rc)))
        end
        JhR = lincomb(terms, parts; kw...)
        s = 1e3 * dt * cv * n * abs(vx)^(n - 1) / (hx * Rc)
        JRhN = s * (-3dvfun(hN)); JRhN1 = s * (dvfun(hN1) / 3)
        JRR = 1e3 * (1 - dt * cv * n * abs(vx)^(n - 1) * vx / Rc)
        x1, l1 = _linsolve(Jhh, -1.0 * rh, -1.0 * rh, st, false)
        x2, l2 = _linsolve(Jhh, JhR, JhR, st, false)
        lin = (max(l1[1], l2[1]), l1[2] + l2[2])
        cx(x) = (v = front_vals(x); JRhN * v[1] + JRhN1 * v[2])
        dR = (-rR - cx(x1)) / (JRR - cx(x2))
        dh = lincomb([1.0, -dR], [x1, x2]; kw...)
        m0 = merit(rh, rR); lam = 1.0; ok = false
        local cand
        for _ in 0:8
            hn = lincomb([1.0, lam], [hc, dh]; kw...); Rn = Rc + lam * dR
            cand = (hn, Rn, residuals(hn, Rn, h, R, dt)...)
            ok = merit(cand[3], cand[4]) <= (1 - 1e-4lam) * m0
            ok && break
            lam /= 2
        end
        ch = norm(dh) / norm(hc) + abs(dR) / Rc
        # no descent: at the truncation-noise floor of the merit take the (tiny) full step as converged
        ok || return (ch < cfg["noise_tol"] ? (lincomb([1.0, 1.0], [hc, dh]; kw...), Rc + dR, true, it, ch, lin, chiJ) :
                      (hc, Rc, false, it, ch, lin, chiJ))
        hc, Rc, rh, rR, F, FD, adv, fv, Rdot = cand
        ch < cfg["tol"] && break
    end
    return hc, Rc, ch < cfg["tol"], it, ch, lin, chiJ
end

println(@sprintf("Landau 1D MPS %s: 2^%d cells in xi, dt = %g a, lin_cutoff %g", cfg["case"], nb, cfg["dt"], cfg["lin_cutoff"]))
t_end = t + cfg["t_run"]; dt = cfg["dt"]; nst = 0; nit = 0; nrej = 0
mass(hc, Rc) = Rc * sum(Array(contract(hc))) * hx
wall = @elapsed while t < t_end - 1e-9
    global h, R, t, dt, nst, nit, nrej
    dtt = min(dt, t_end - t)
    ts = @elapsed (hn, Rn, ok, it, ch, lin, chiJ) = be_step(h, R, dtt)
    if !ok
        nrej += 1; dt /= 2
        println(@sprintf("  rejected (it %d, change %.1e, lin res %.1e), dt -> %g", it, ch, lin[1], dt)); continue
    end
    h, R, t = hn, Rn, t + dtt; nst += 1; nit += it
    dt = min(cfg["dt"], 1.5dt)
    if cfg["report"] > 0 && nst % cfg["report"] == 0
        println(@sprintf("t = %9.1f a  dt %7.1f  R = %.4f km  H(0) = %.2f m  newton %d  lin res %.1e  chi(h) %d (x%.0f)  chi(J) %d  %.1f s",
                         t, dtt, R / 1e3, MPSEvaluator(h)(bits(0)), it, lin[1], maxlinkdim(h), compression(h, g), chiJ, ts))
        flush(stdout)
    end
end
@printf("%d steps (%d rejected), mean Newton %.1f, %.1f s; chi(h) %d, %d numbers for %d cells (x%.0f)\n", nst, nrej, nit / max(nst, 1), wall,
        maxlinkdim(h), nparams(h), N, compression(h, g))
Eh = MPSEvaluator(h); xs = [(i - 0.5) * hx for i in 1:N]
sample = unique(round.(Int, range(1, N, length = min(N, 4096))))
if cfg["case"] == "halfar"
    Re = halfar_margin(hb, t)
    err = maximum(abs(Eh(bits(i - 1)) - halfar_H(hb, t, xs[i] * Re)) for i in sample)
    @printf("t = %.0f a: margin %.4f km (exact %.4f, error %.3f km = %.2f cells of %.3f km), dome %.3f m (exact %.3f), max |H - H_exact| %.3f m (sampled)\n",
            t, R / 1e3, Re / 1e3, (R - Re) / 1e3, (R - Re) / (Re * hx), Re * hx / 1e3, Eh(bits(0)), halfar_H(hb, t, xs[1] * Re), err)
else
    @printf("t = %.0f a: margin %.4f km (closure %.4f km), dome %.3f m\n", t, R / 1e3, eismint_margin(eb, 1) / 1e3, Eh(bits(0)))
end
