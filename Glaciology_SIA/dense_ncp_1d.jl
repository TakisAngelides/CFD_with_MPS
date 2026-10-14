# Dense (sparse-matrix) replica of the MPS EISMINT time loop: variable-step BDF2, adaptive dt
# (halve on failure, x dt_grow after easy steps) and the complementarity formulation of H >= 0,
# to tell algorithmic Newton failures from MPS precision effects. Complementarity functions:
#   ssn  Phi = m R + (1 - m) H, m = 1[H >= c R]   (semismooth Newton on min(H, c R), as in sia.jl)
#   fb   Phi = H + c R - sqrt(H^2 + (c R)^2)      (Fischer-Burmeister, C^1 merit ||Phi||^2)
#   sfb  smoothed FB, sqrt(H^2 + (c R)^2 + 2 mu^2) with mu = mu_rel * dt * max|a|
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch dense_ncp_1d.jl nbits=10 ncp=ssn
using LinearAlgebra, SparseArrays, Printf, HDF5
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

cfg = Dict{String,Any}("nbits" => 10, "dt" => 1000.0, "t_run" => 80000.0, "ncp" => "ssn", "c" => 1.0, "mu_rel" => 1e-3,
                       "max_iters" => 10, "tol" => 1e-5, "line_search" => 6, "dt_grow" => 1.5, "grow_iters" => 6,
                       "max_halvings" => 8, "noise_tol" => 1e-4, "init" => "", "log" => false, "report_every" => 10000.0)
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : cfg[k] isa Bool ? parse(Bool, v) : parse(typeof(cfg[k]), v)
end

p = SIAParams()
eb = EISMINTMovingMargin(p = p)
N = 2^cfg["nbits"]; L = eb.L; h = 2L / N
x = [-L + (i + 0.5) * h for i in 0:N-1]
a = [eismint_mb(eb, abs(xi)) for xi in x]
Sp = spdiagm(1 => ones(N - 1))
Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
G = Gamma(p)
Dface(H) = (He = Avg * H; s = Df * H; G .* He .^ 5 .* s .^ 2)
F(H) = Db * (Dface(H) .* (Df * H))
Jac(H) = (He = Avg * H; g = Df * H;
          Db * (spdiagm(0 => 3G .* He .^ 5 .* g .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* g .^ 3) * Avg))

c = cfg["c"]
# complementarity function and its (generalised) Jacobian, given R and dR/dH
function ncp_fun(H, R, dR, mu)
    if cfg["ncp"] == "ssn"
        m = H .>= c .* R
        return ifelse.(m, c .* R, H), spdiagm(0 => Float64.(m)) * (c * dR) + spdiagm(0 => Float64.(.!m))
    else
        cR = c .* R
        rho = sqrt.(H .^ 2 .+ cR .^ 2 .+ 2mu^2)
        rho0 = max.(rho, eps())
        da = 1 .- H ./ rho0; db = 1 .- cR ./ rho0
        z = rho .== 0                       # FB kink (only at mu = 0): any element of the B-subdifferential
        da[z] .= 1 - 1 / sqrt(2); db[z] .= 1 - 1 / sqrt(2)
        return H .+ cR .- rho, spdiagm(0 => da) + spdiagm(0 => db) * (c * dR)
    end
end

function step!(H, Hp, dt, omega)
    if Hp === nothing
        hist, dte = H, dt
    else
        w = omega
        hist = ((1 + w)^2 * H - w^2 * Hp) / (1 + 2w); dte = dt * (1 + w) / (1 + 2w)
    end
    rhs0 = hist + dte * a
    mu = cfg["ncp"] == "sfb" ? cfg["mu_rel"] * dte * maximum(abs, a) : 0.0
    Res(Hc) = Hc - dte * F(Hc) - rhs0
    merit(Hc) = norm(ncp_fun(Hc, Res(Hc), I, mu)[1])
    Hn = copy(H); change = Inf; it = 0; fails = 0
    for m in 1:cfg["max_iters"]
        it = m
        R = Res(Hn)
        Phi, J = ncp_fun(Hn, R, I - dte * Jac(Hn), mu)
        nR = norm(Phi)
        d = try lu(J) \ (-Phi) catch; qr(Matrix(J)) \ (-Phi) end
        change = norm(d) / norm(Hn)
        lam = 1.0; ok = false
        for _ in 0:cfg["line_search"]
            ok = merit(Hn + lam * d) <= (1 - 1e-4lam) * nR
            ok && break
            lam /= 2
        end
        cfg["log"] && println(@sprintf("    it %2d |Phi|/|rhs| %.2e |d|/|H| %.2e lambda %s  active %d", m, nR / norm(rhs0), change,
                                       ok ? @sprintf("%.3g", lam) : "fail", count(Hn .< c .* R)))
        # no descent: at the round-off floor of the merit (steady state) take the full step, as sia.jl
        ok || return (change < cfg["noise_tol"] ? (max.(Hn + d, 0.0), true, it, 0.0) : (Hn, false, it, change))
        Hn = Hn + lam * d
        change < cfg["tol"] && break
    end
    return max.(Hn, 0.0), change < cfg["tol"], it, change
end

if isempty(cfg["init"])
    H = max.(a, 0.0) .* cfg["dt"]; t = 0.0
else                                      # start from the final state of an MPS run on the same grid
    Hm, nb, ordm, t = h5open(cfg["init"], "r") do f
        read(f, "H_mps", MPS), read(attributes(f)["nbits"]), Symbol(read(attributes(f)["ordering"])), read(f["times"])[end]
    end
    nb <= cfg["nbits"] || error("init file has 2^$(nb) points > 2^$(cfg["nbits"])")
    gq = QuanticsGrid(nb, ordm, (-L, L))
    gf = QuanticsGrid(cfg["nbits"], ordm, (-L, L))
    Hm = prolong_mps(replace_siteinds(Hm, gq.sites), gq, gf)      # exact MPS prolongation, as run_sia.jl
    H = max.(vec(mps_to_array(Hm, gf)), 0.0)
end
Hp = nothing
t_end = t + cfg["t_run"]; next_report = t + cfg["report_every"]; dt_cur = cfg["dt"]; dt_prev = nothing; dt_min = cfg["dt"] / 2^cfg["max_halvings"]
nstep = 0; nrej = 0; win_rej = 0; win_steps = 0; win_iters = 0
margin(H) = (i = findlast(>(1.0), H); i === nothing ? 0.0 : x[i])
println(@sprintf("dense %s (c = %g): 2^%d points, dx = %.2f km, dt = %g a, closure margin %.1f km", cfg["ncp"], c, cfg["nbits"], h / 1e3,
                 cfg["dt"], eismint_margin(eb, 1) / 1e3))
wall = @elapsed while t < t_end - 1e-6
    global H, Hp, t, dt_cur, dt_prev, nstep, nrej, win_rej, win_steps, win_iters, next_report
    local Hn, ok, it, ch, dt_try
    while true
        dt_try = min(dt_cur, t_end - t)
        omega = dt_prev === nothing ? 1.0 : dt_try / dt_prev
        Hn, ok, it, ch = step!(H, Hp, dt_try, omega)
        (ok || dt_try <= dt_min * (1 + 1e-9)) && break
        nrej += 1; win_rej += 1
        dt_cur = dt_try / 2
        cfg["log"] && println(@sprintf("  rejected at t = %.0f (it %d, |d|/|H| %.1e) dt -> %.4g", t, it, ch, dt_cur))
    end
    nstep += 1; win_steps += 1; win_iters += it
    t += dt_try; Hp, H = H, Hn; dt_prev = dt_try
    ok && it <= cfg["grow_iters"] && (dt_cur = min(cfg["dt"], dt_cur * cfg["dt_grow"]))
    if t >= next_report - 1e-6 || t >= t_end - 1e-6
        println(@sprintf("t = %6.0f a  steps %3d  rejected %3d  mean newton %.1f  mean dt %6.1f a  H_ctr %.1f m  margin %.1f km",
                         t, win_steps, win_rej, win_iters / win_steps, cfg["report_every"] / win_steps, H[N ÷ 2 + 1], margin(H) / 1e3))
        flush(stdout)
        win_rej = win_steps = win_iters = 0; next_report += cfg["report_every"]
    end
end
im = findlast(>(0.0), H)
println("cells at the +x margin (x km => H m): ", join([@sprintf("%.2f => %.2f", x[i] / 1e3, H[i]) for i in max(1, im - 4):min(N, im + 1)], ", "))
println(@sprintf("total: %d steps, %d rejected, %.1f s", nstep, nrej, wall))
