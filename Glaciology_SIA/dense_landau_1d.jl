# 1D SIA in front-fixing (Landau) coordinates, dense sparse-Newton reference before the MPS version.
# x = xi R(t), xi in [0, 1] (symmetric at xi = 0, margin at xi = 1), unknowns h(xi) and R.
# Conservative form (m = R h is the mass per unit xi):
#   d/dt (R h) = d/dxi (xi Rdot h) - d/dxi q + R a(xi R),   q = -Gamma h^(n+2) |h_x|^(n-1) h_x,  h_x = h_xi / R
# Zero flux through xi = 1 (h = 0 there): the margin moves with the ice, so at steady state the total
# mass balance over [0, R] vanishes and R is the mass-balance closure. Margin speed = ice velocity at
# the margin, from the Lipschitz "pressure" v = h^((2n+1)/n):  Rdot = -Gamma (n/(2n+1))^n |v_x|^(n-1) v_x.
# Finite volumes in xi (Mahaffy faces, upwind advection), backward Euler for (h, R) together; Newton
# with an exact complex-step Jacobian. No complementarity, no margin cells: the margin is not limited
# to ~1 cell per Newton iteration, so dt is not tied to the grid spacing.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch dense_landau_1d.jl case=halfar nbits=10 dt=100
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

cfg = Dict{String,Any}("case" => "halfar", "nbits" => 10, "dt" => 100.0, "t_run" => 20000.0, "tol" => 1e-10, "max_iters" => 60,
                       "report" => 0, "split" => false, "R_init" => 300e3, "H_init" => 1000.0)
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end
p = SIAParams(); n = p.n_glen; G = Gamma(p)
N = 2^cfg["nbits"]; hx = 1 / N
xi = [(i - 0.5) * hx for i in 1:N]; xf = [j * hx for j in 0:N]          # cell centres, faces 0..N
if cfg["case"] == "halfar"
    hb = Halfar(d = 1, p = p); t0 = halfar_t0(hb)
    amb(x) = 0.0
    R = halfar_margin(hb, t0); h = [halfar_H(hb, t0, x * R) for x in xi]; t = t0
else
    eb = EISMINTMovingMargin(p = p)
    amb(x) = eismint_mb(eb, abs(x))
    R = cfg["R_init"]; h = [cfg["H_init"] * sqrt(max(1 - x^2, 0.0)) for x in xi]; t = 0.0
end
cv = G * (n / (2n + 1))^n

# residual of a backward Euler step for z = [h; R]
function residual(z, hold, Rold, dt)
    T = eltype(z); hh = view(z, 1:N); Rn = z[N+1]
    Rdot = (Rn - Rold) / dt
    F = zeros(T, N + 1)                       # total flux q - xi Rdot h on faces 0..N (both ends zero)
    for j in 1:N-1
        Hf = (hh[j] + hh[j+1]) / 2; g = (hh[j+1] - hh[j]) / (hx * Rn)
        q = -G * Hf^(n + 2) * (g^2)^div(n - 1, 2) * g
        hup = real(Rdot) < 0 ? hh[j] : hh[j+1]          # transport velocity -xi Rdot in xi
        F[j+1] = q - xf[j+1] * Rdot * hup
    end
    r = similar(z)
    for i in 1:N
        r[i] = Rn * hh[i] - Rold * hold[i] + dt * ((F[i+1] - F[i]) / hx - Rn * amb(xi[i] * Rn))
    end
    # margin speed: v = h |h|^((n+1)/n), one-sided 2nd-order derivative at xi = 1 through v(1) = 0
    v(y) = y * (y^2)^((n + 1) / (2n))
    vx = (-3v(hh[N]) + v(hh[N-1]) / 3) / (hx * Rn)
    r[N+1] = (Rn - Rold) + dt * cv * (vx^2)^div(n - 1, 2) * vx
    r[N+1] *= 1e3                              # front row [m] -> [m^2] like the cell rows (R h)
    return r
end
function jacobian(z, hold, Rold, dt)
    del = 1e-30; Ii = Int[]; Jj = Int[]; V = Float64[]
    for c in 0:2
        e = zeros(N + 1); e[(c+1):3:N] .= 1
        dr = imag.(residual(complex.(z, del .* e), hold, Rold, dt)) ./ del
        for j in (c+1):3:N
            for i in (j-1):(j+1)
                1 <= i <= N && (push!(Ii, i); push!(Jj, j); push!(V, dr[i]))
            end
            j >= N - 1 && (push!(Ii, N + 1); push!(Jj, j); push!(V, dr[N+1]))
        end
    end
    e = zeros(N + 1); e[N+1] = 1
    dr = imag.(residual(complex.(z, del .* e), hold, Rold, dt)) ./ del
    for i in 1:N+1
        push!(Ii, i); push!(Jj, N + 1); push!(V, dr[i])
    end
    return sparse(Ii, Jj, V, N + 1, N + 1)
end
# block Gauss-Seidel Newton (the 2D plan, where R(theta) has N_theta unknowns): h-update with R
# frozen (one linear solve with J_hh), then the margin row alone (scalar per theta)
function be_step_split(h, R, dt)
    z = [h; R]; ch = Inf
    for it in 1:cfg["max_iters"]
        r = residual(z, h, R, dt); J = jacobian(z, h, R, dt)
        dh = J[1:N, 1:N] \ (-r[1:N])
        lam = 1.0; nr = norm(r)
        for _ in 1:10
            norm(residual([z[1:N] + lam * dh; z[N+1]], h, R, dt)) <= (1 - 1e-4lam) * nr && break
            lam /= 2
        end
        z[1:N] .+= lam .* dh
        r = residual(z, h, R, dt); J = jacobian(z, h, R, dt)
        dR = -r[N+1] / J[N+1, N+1]
        z[N+1] += dR
        ch = norm(lam * dh) / norm(z[1:N]) + abs(dR) / z[N+1]
        ch < cfg["tol"] && return z[1:N], z[N+1], true, it
    end
    return z[1:N], z[N+1], ch < 1e-6, cfg["max_iters"]
end

function be_step(h, R, dt)
    cfg["split"] && return be_step_split(h, R, dt)
    z = [h; R]; ch = Inf
    for it in 1:cfg["max_iters"]
        r = residual(z, h, R, dt); nr = norm(r)
        d = jacobian(z, h, R, dt) \ (-r)
        lam = 1.0
        for _ in 1:10
            norm(residual(z + lam * d, h, R, dt)) <= (1 - 1e-4lam) * nr && break
            lam /= 2
        end
        z = z + lam * d
        ch = norm(d[1:N]) / norm(z[1:N]) + abs(d[N+1]) / z[N+1]
        ch < cfg["tol"] && return z[1:N], z[N+1], true, it
    end
    return z[1:N], z[N+1], ch < 1e-6, cfg["max_iters"]
end

println(@sprintf("Landau 1D %s: 2^%d cells in xi, dt = %g a", cfg["case"], cfg["nbits"], cfg["dt"]))
t_end = t + cfg["t_run"]; dt = cfg["dt"]; nst = 0; nit = 0; nrej = 0
mass(h, R) = R * sum(h) * hx
M0 = mass(h, R)
wall = @elapsed while t < t_end - 1e-9
    global h, R, t, dt, nst, nit, nrej
    dtt = min(dt, t_end - t)
    hn, Rn, ok, it = be_step(h, R, dtt)
    if !ok
        nrej += 1; dt /= 2; continue
    end
    h, R, t = hn, Rn, t + dtt; nst += 1; nit += it
    dt = min(cfg["dt"], 1.5dt)
    if cfg["report"] > 0 && nst % cfg["report"] == 0
        println(@sprintf("t = %9.1f a  R = %.3f km  H(0) = %.2f m  newton %d", t, R / 1e3, h[1], it))
    end
end
@printf("%d steps (%d rejected), mean Newton %.1f, %.1f s\n", nst, nrej, nit / max(nst, 1), wall)
if cfg["case"] == "halfar"
    Re = halfar_margin(hb, t); He = [halfar_H(hb, t, x * Re) for x in xi]
    # error on the exact solution's own nodes (same xi), and the dome
    @printf("t = %.0f a: margin %.4f km (exact %.4f, error %.3f km = %.2f cells of %.3f km), dome %.3f m (exact %.3f), max |H - H_exact| %.3f m, mass change %.1e\n",
            t, R / 1e3, Re / 1e3, (R - Re) / 1e3, (R - Re) / (Re * hx), Re * hx / 1e3, h[1], He[1], maximum(abs.(h .- He)), mass(h, R) / M0 - 1)
else
    @printf("t = %.0f a: margin %.4f km (closure %.4f km), dome %.3f m, min h %.3e\n", t, R / 1e3, eismint_margin(eb, 1) / 1e3, h[1], minimum(h))
end
