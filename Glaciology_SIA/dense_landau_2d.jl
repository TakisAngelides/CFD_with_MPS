# 2D SIA in front-fixing (Landau) coordinates, dense sparse-Newton reference: x = xi R(theta, t) e_r,
# xi in [0, 1] (pole at 0, margin at 1), theta periodic. Unknowns h(xi, theta) and R(theta).
# Metric (r = xi R): J = xi R^2, grad xi = (e_r - (R'/R) e_theta)/R, grad theta = e_theta/(xi R),
#   g^xx = (1 + R'^2/R^2)/R^2, g^xt = -R'/(xi R^3), g^tt = 1/(xi^2 R^2)
# Conservative moving-grid form (grid velocity xi Rdot e_r):
#   d/dt (xi R^2 h) + d/dxi (JQ^xi) + d/dtheta (JQ^th) = xi R^2 a
#   JQ^xi = -D [xi (1 + R'^2/R^2) h_xi - (R'/R) h_th] - xi^2 R Rdot h,   JQ^th = -D [-(R'/R) h_xi + h_th/xi]
#   D = Gamma h^(n+2) |grad h|^(n-1),  |grad h|^2 = g^xx h_xi^2 + 2 g^xt h_xi h_th + g^tt h_th^2
# Zero flux through the margin face; margin speed = normal ice velocity:
#   Rdot = -c_v |v_xi|^(n-1) v_xi |grad xi|^(n+1) R,  v = h |h|^((n+1)/n)
# Backward Euler for (h, R) together, Newton with an exact complex-step Jacobian (12 colours), sparse LU.
#   sbatch --time=02:00:00 slurm/run.sbatch dense_landau_2d.jl case=halfar nbits=6 dt=100
using LinearAlgebra, SparseArrays, Printf, HDF5
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

cfg = Dict{String,Any}("case" => "halfar", "nbits" => 6, "nth" => 0, "dt" => 100.0, "t_run" => 20000.0, "tol" => 1e-9, "max_iters" => 20,
                       "report" => 5, "R_init" => 300e3, "H_init" => 1000.0, "ell" => 0.2, "noise_tol" => 1e-7, "steady_tol" => 0.0,
                       "cutoffs" => "1e-8,1e-12", "init" => "")
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end
p = SIAParams(); n = p.n_glen; G = Gamma(p); cv = G * (n / (2n + 1))^n
Nx = 2^cfg["nbits"]; Nt = cfg["nth"] > 0 ? cfg["nth"] : Nx
hx = 1 / Nx; ht = 2pi / Nt
xi = [(i - 0.5) * hx for i in 1:Nx]; th = [(j - 1) * ht for j in 1:Nt]
ell = cfg["ell"]
if cfg["case"] == "halfar"
    hb = Halfar(d = 2, p = p); t = halfar_t0(hb)
    amb(r, tt) = 0.0
    R = fill(halfar_margin(hb, t), Nt); h = [halfar_H(hb, t, x * R[j]) for x in xi, j in 1:Nt]
else
    eb = EISMINTMovingMargin(p = p)
    # "eismint": radial mass balance (circular closure margin); "ellipse": the ELA on an ellipse
    mbc(r) = (v = eb.S * (eb.R_el - r); real(v) < eb.Mmax ? v : eb.Mmax + 0 * v)     # eismint_mb, complex-step safe
    amb(r, tt) = cfg["case"] == "eismint" ? mbc(r) : mbc(r * sqrt((cos(tt) / (1 + ell))^2 + (sin(tt) / (1 - ell))^2))
    R = fill(cfg["R_init"], Nt); h = [cfg["H_init"] * sqrt(max(1 - x^2, 0.0)) for x in xi, j in 1:Nt]; t = 0.0
end
if cfg["init"] != ""
    h, R, t = h5open(f -> (read(f, "h"), read(f, "R"), read(f, "t")), cfg["init"])
    size(h) == (Nx, Nt) || error("init grid mismatch")
    println(@sprintf("init from %s: t = %.0f a", cfg["init"], t))
end
vfun(y) = y * (y^2)^((n + 1) / (2n))
nunk = Nx * Nt + Nt
li = LinearIndices((Nx, Nt))

function residual(z, hold, Rold, dt)
    T = eltype(z)
    hh = reshape(view(z, 1:Nx*Nt), Nx, Nt); Rv = view(z, Nx*Nt+1:nunk)
    Rdot = (Rv .- Rold) ./ dt
    # ghost-extended h: i = 0 (pole, zero gradient), i = Nx + 1 (reflection: h = 0 at xi = 1); theta periodic
    hg(i, j) = (jj = mod1(j, Nt); i < 1 ? hh[1, jj] : i > Nx ? -hh[Nx, jj] : hh[i, jj])
    Rp_c = [(Rv[mod1(j + 1, Nt)] - Rv[mod1(j - 1, Nt)]) / (2ht) for j in 1:Nt]
    # xi-faces f = 1..Nx-1 (between i = f and f + 1); faces 0 and Nx carry no flux
    FX = zeros(T, Nx + 1, Nt)
    for j in 1:Nt, f in 1:Nx-1
        x = f * hx; Rj = Rv[j]; Rp = Rp_c[j]
        H = (hh[f, j] + hh[f+1, j]) / 2
        hxi = (hh[f+1, j] - hh[f, j]) / hx
        hth = (hg(f, j + 1) - hg(f, j - 1) + hg(f + 1, j + 1) - hg(f + 1, j - 1)) / (4ht)
        g2 = (1 + (Rp / Rj)^2) / Rj^2 * hxi^2 - 2Rp / (x * Rj^3) * hxi * hth + hth^2 / (x * Rj)^2
        D = G * H^(n + 2) * g2^div(n - 1, 2)
        hup = real(Rdot[j]) > 0 ? hh[f+1, j] : hh[f, j]
        FX[f+1, j] = -D * (x * (1 + (Rp / Rj)^2) * hxi - Rp / Rj * hth) - x^2 * Rj * Rdot[j] * hup
    end
    # theta-faces j + 1/2 (between j and j + 1)
    FT = zeros(T, Nx, Nt)
    for j in 1:Nt, i in 1:Nx
        jp = mod1(j + 1, Nt); x = xi[i]
        Rf = (Rv[j] + Rv[jp]) / 2; Rp = (Rv[jp] - Rv[j]) / ht
        H = (hh[i, j] + hh[i, jp]) / 2
        hth = (hh[i, jp] - hh[i, j]) / ht
        hxi = (hg(i + 1, j) - hg(i - 1, j) + hg(i + 1, jp) - hg(i - 1, jp)) / (4hx)
        g2 = (1 + (Rp / Rf)^2) / Rf^2 * hxi^2 - 2Rp / (x * Rf^3) * hxi * hth + hth^2 / (x * Rf)^2
        D = G * H^(n + 2) * g2^div(n - 1, 2)
        FT[i, j] = -D * (-(Rp / Rf) * hxi + hth / x)
    end
    r = similar(z)
    for j in 1:Nt, i in 1:Nx
        Rj = Rv[j]; x = xi[i]
        r[li[i, j]] = x * Rj^2 * hh[i, j] - x * Rold[j]^2 * hold[i, j] +
                      dt * ((FX[i+1, j] - FX[i, j]) / hx + (FT[i, j] - FT[i, mod1(j - 1, Nt)]) / ht - x * Rj^2 * amb(x * Rj, th[j]))
    end
    for j in 1:Nt
        Rj = Rv[j]; Rp = Rp_c[j]
        vxi = (-3vfun(hh[Nx, j]) + vfun(hh[Nx-1, j]) / 3) / hx
        gx = sqrt(1 + (Rp / Rj)^2) / Rj
        r[Nx*Nt+j] = 1e3 * hx * Rj * (Rj - Rold[j] + dt * cv * (vxi^2)^div(n - 1, 2) * vxi * gx^(n + 1) * Rj)
    end
    return r
end

# colours: h by (i mod 3, j mod 4), R by j mod 4 (theta periodic with Nt a multiple of 4, stencil radius 1)
function jacobian(z, hold, Rold, dt)
    Nt % 4 == 0 || error("Nt must be a multiple of 4")
    del = 1e-30; Ii = Int[]; Jj = Int[]; V = Float64[]
    for ci in 0:2, cj in 0:3
        e = zeros(nunk); E = reshape(view(e, 1:Nx*Nt), Nx, Nt); E[ci+1:3:end, cj+1:4:end] .= 1
        dr = imag.(residual(complex.(z, del .* e), hold, Rold, dt)) ./ del
        for j in cj+1:4:Nt, i in ci+1:3:Nx
            col = li[i, j]
            for dj in -1:1, di in -1:1
                r_ = i + di; s_ = mod1(j + dj, Nt)
                1 <= r_ <= Nx && (push!(Ii, li[r_, s_]); push!(Jj, col); push!(V, dr[li[r_, s_]]))
            end
            i >= Nx - 1 && (push!(Ii, Nx * Nt + j); push!(Jj, col); push!(V, dr[Nx*Nt+j]))
        end
    end
    for cj in 0:3
        e = zeros(nunk); js = collect(cj+1:4:Nt); e[Nx*Nt .+ js] .= 1
        dr = imag.(residual(complex.(z, del .* e), hold, Rold, dt)) ./ del
        for j in js, dj in -1:1
            s_ = mod1(j + dj, Nt)
            for i in 1:Nx
                push!(Ii, li[i, s_]); push!(Jj, Nx * Nt + j); push!(V, dr[li[i, s_]])
            end
            push!(Ii, Nx * Nt + s_); push!(Jj, Nx * Nt + j); push!(V, dr[Nx*Nt+s_])
        end
    end
    return sparse(Ii, Jj, V, nunk, nunk)
end

function be_step(h, R, dt)
    z = [vec(h); R]; ch = Inf
    for it in 1:cfg["max_iters"]
        r = residual(z, h, R, dt); nr = norm(r)
        d = lu(jacobian(z, h, R, dt)) \ (-r)
        lam = 1.0; ok = false
        for _ in 1:10
            ok = norm(residual(z + lam * d, h, R, dt)) <= (1 - 1e-4lam) * nr; ok && break
            lam /= 2
        end
        ch = norm(d[1:Nx*Nt]) / norm(z[1:Nx*Nt]) + norm(d[Nx*Nt+1:end]) / norm(z[Nx*Nt+1:end])
        if !ok
            ch < cfg["noise_tol"] && (z = z + d; return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], true, it)
            return h, R, false, it
        end
        z = z + lam * d
        ch < cfg["tol"] && return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], true, it
    end
    return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], ch < 1e-6, cfg["max_iters"]
end

println(@sprintf("Landau 2D %s: %d x %d (xi x theta), dt = %g a", cfg["case"], Nx, Nt, cfg["dt"]))
cfg["case"] != "halfar" && println(@sprintf("closure margin (radial EISMINT) %.2f km", eismint_margin(eb, 2) / 1e3))
mass(h, R) = sum(xi[i] * R[j]^2 * h[i, j] for i in 1:Nx, j in 1:Nt) * hx * ht
M0 = mass(h, R)
t_end = t + cfg["t_run"]; dt = cfg["dt"]; nst = 0; nit = 0; nrej = 0
wall = @elapsed while t < t_end - 1e-9
    global h, R, t, dt, nst, nit, nrej
    dtt = min(dt, t_end - t)
    hn, Rn, ok, it = be_step(h, R, dtt)
    if !ok
        nrej += 1; dt /= 2; dt < 1e-4 && error("dt collapsed"); continue
    end
    rate = norm(Rn - R) / norm(R) / dtt
    h, R, t = hn, Rn, t + dtt; nst += 1; nit += it
    it <= 6 && (dt = min(cfg["dt"], 1.5dt))
    nst % cfg["report"] == 0 && println(@sprintf("t = %9.1f a  dt %7.1f  R in [%.3f, %.3f] km  h(pole) %.2f m  newton %d  |dR/dt|/|R| %.1e",
                                                 t, dtt, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1], it, rate))
    flush(stdout)
    cfg["steady_tol"] > 0 && rate < cfg["steady_tol"] && break
end
@printf("%d steps (%d rejected), mean Newton %.1f, %.1f s; mass change %.1e (halfar) \n", nst, nrej, nit / max(nst, 1), wall, mass(h, R) / M0 - 1)
if cfg["case"] == "halfar"
    Re = halfar_margin(hb, t)
    He = [halfar_H(hb, t, x * Re) for x in xi]
    @printf("t = %.0f a: margin %.4f .. %.4f km (exact %.4f), pole %.3f m (exact %.3f), max |h - h_exact| %.3f m\n",
            t, minimum(R) / 1e3, maximum(R) / 1e3, Re / 1e3, h[1, 1], He[1], maximum(abs.(h .- He)))
else
    @printf("t = %.0f a: margin %.3f .. %.3f km, h(pole) %.2f m\n", t, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1])
end
mkpath("output"); fn = cfg["init"] == "" ? "output/landau2d_$(cfg["case"])_n$(cfg["nbits"]).h5" : "output/landau2d_$(cfg["case"])_n$(cfg["nbits"])_restart.h5"
h5open(fn, "w") do f
    f["h"] = h; f["R"] = R; f["t"] = t
end
if Nt == Nx
    g = QuanticsGrid(cfg["nbits"], :interleaved, (0.0, 1.0), (0.0, 2pi))
    for c in parse.(Float64, split(cfg["cutoffs"], ","))
        @printf("cutoff %.0e: chi(h) %d (x%.0f)\n", c, maxlinkdim(array_to_mps(h, g; cutoff = c)), Nx * Nt / nparams(array_to_mps(h, g; cutoff = c)))
    end
end
