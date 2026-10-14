# Callable version of Glaciology_SIA/dense_landau_2d.jl (dense sparse-Newton 2D SIA in front-fixing
# (Landau) coordinates x = xi R(theta, t) e_r), for parameter studies. Same discretisation, residual,
# complex-step Jacobian, Newton and line search as the original script; differences:
#   * everything lives in a function (type-stable, thread-safe: no globals), Jacobian sparsity pattern
#     and the symbolic LU are reused;
#   * the EISMINT-ellipse mass balance gets an extra asymmetry Mmax(theta) = Mmax (1 + asym cos theta)
#     (asym = 0 reproduces the original exactly) and Glen's A is a parameter;
#   * a fixed-dt transient phase to t_out (fixed time grid -> the QoI is a smooth function of the
#     parameters; a failing step is split recursively in halves and counted), then a steady-state phase
#     with geometrically growing dt (backward Euler with dt -> infinity is Newton on the steady problem).
# Physics constants as Glaciology_SIA/sia.jl (SIAParams: n = 3, A = 1e-16 Pa^-3 a^-1, rho = 910, g = 9.81).
using LinearAlgebra, SparseArrays, Printf

const NGLEN = 3
const RHO_ICE = 910.0
const GRAV = 9.81
const A_REF = 1e-16

Base.@kwdef struct ModelParams
    Afac::Float64 = 1.0        # Glen A = Afac * 1e-16 Pa^-3 a^-1
    S::Float64 = 1e-5          # mass-balance gradient (a^-1)
    R_el::Float64 = 450e3      # equilibrium-line radius (m)
    Mmax::Float64 = 0.5        # maximum accumulation (m/a)
    ell::Float64 = 0.2         # ellipticity of the ELA
    asym::Float64 = 0.0        # Mmax(theta) = Mmax (1 + asym cos theta)
    R_init::Float64 = 300e3    # initial ice-cap radius (m)
    H_init::Float64 = 1000.0   # initial dome thickness (m)
end

struct LandauModel
    Nx::Int
    Nt::Int
    hx::Float64
    ht::Float64
    xi::Vector{Float64}
    th::Vector{Float64}
    G::Float64          # Gamma = 2 A (rho g)^n / (n + 2)
    cv::Float64
    S::Float64
    R_el::Float64
    mmax::Vector{Float64}   # Mmax(theta_j)
    efac::Vector{Float64}   # elliptic radius factor at theta_j
    eismint::Bool           # radial mass balance (case = eismint of the original)
    nunk::Int
    tol::Float64
    max_iters::Int
    noise_tol::Float64
end

function LandauModel(par::ModelParams; nbits::Int = 5, nth::Int = 0, tol = 1e-9, max_iters = 20, noise_tol = 1e-7,
                     eismint::Bool = false)
    Nx = 2^nbits; Nt = nth > 0 ? nth : Nx
    hx = 1 / Nx; ht = 2pi / Nt
    xi = [(i - 0.5) * hx for i in 1:Nx]; th = [(j - 1) * ht for j in 1:Nt]
    n = NGLEN
    G = 2 * (A_REF * par.Afac) * (RHO_ICE * GRAV)^n / (n + 2)
    cv = G * (n / (2n + 1))^n
    ell = par.ell
    efac = [sqrt((cos(t) / (1 + ell))^2 + (sin(t) / (1 - ell))^2) for t in th]
    mmax = [par.Mmax * (1 + par.asym * cos(t)) for t in th]
    return LandauModel(Nx, Nt, hx, ht, xi, th, G, cv, par.S, par.R_el, mmax, efac, eismint, Nx * Nt + Nt,
                       tol, max_iters, noise_tol)
end

# a = min(Mmax, S (R_el - r_eff)), complex-step safe (as mbc in the original)
@inline function amb(m::LandauModel, r, j::Int)
    reff = m.eismint ? r : r * m.efac[j]
    v = m.S * (m.R_el - reff)
    return real(v) < m.mmax[j] ? v : m.mmax[j] + 0 * v
end

@inline vfun(y) = y * (y^2)^((NGLEN + 1) / (2NGLEN))

# Work buffers of the residual (one set per element type: Float64 for residuals, ComplexF64 for the
# complex-step Jacobian), so that a Newton iteration does not allocate (GC contention with many threads).
struct ResBuf{T}
    FX::Matrix{T}
    FT::Matrix{T}
    Rp::Vector{T}
    Rdot::Vector{T}
    r::Vector{T}
end
ResBuf{T}(m) where {T} = ResBuf{T}(zeros(T, m.Nx + 1, m.Nt), zeros(T, m.Nx, m.Nt), zeros(T, m.Nt), zeros(T, m.Nt), zeros(T, m.nunk))

residual(m::LandauModel, z::AbstractVector{T}, hold, Rold, dt) where {T} = copy(residual!(ResBuf{T}(m), m, z, hold, Rold, dt))

# steady = true: the steady equations, i.e. the backward-Euler residual / dt with dt -> infinity (no storage,
# no grid velocity) and the margin condition Rdot = 0 written as v_xi = 0. The latter is equivalent but not
# degenerate: Rdot ~ v_xi^n has a zero derivative at a steady margin (n = 3), which is why time stepping (and
# Newton on the BE residual with huge dt) only approaches the steady state algebraically.
function residual!(buf::ResBuf{T}, m::LandauModel, z::AbstractVector{T}, hold, Rold, dt; steady::Bool = false) where {T}
    Nx, Nt, hx, ht, xi, G, cv = m.Nx, m.Nt, m.hx, m.ht, m.xi, m.G, m.cv
    n = NGLEN
    hh = reshape(view(z, 1:Nx*Nt), Nx, Nt); Rv = view(z, Nx*Nt+1:m.nunk)
    Rdot = buf.Rdot; Rp_c = buf.Rp; FX = buf.FX; FT = buf.FT; r = buf.r
    @inbounds for j in 1:Nt
        Rdot[j] = steady ? zero(T) : (Rv[j] - Rold[j]) / dt
        Rp_c[j] = (Rv[mod1(j + 1, Nt)] - Rv[mod1(j - 1, Nt)]) / (2ht)
    end
    @inline hg(i, j) = (jj = mod1(j, Nt); i < 1 ? hh[1, jj] : i > Nx ? -hh[Nx, jj] : hh[i, jj])
    # FX[1, :] and FX[Nx+1, :] (faces 0 and Nx) carry no flux and stay zero
    @inbounds for j in 1:Nt, f in 1:Nx-1
        x = f * hx; Rj = Rv[j]; Rp = Rp_c[j]
        H = (hh[f, j] + hh[f+1, j]) / 2
        hxi = (hh[f+1, j] - hh[f, j]) / hx
        hth = (hg(f, j + 1) - hg(f, j - 1) + hg(f + 1, j + 1) - hg(f + 1, j - 1)) / (4ht)
        g2 = (1 + (Rp / Rj)^2) / Rj^2 * hxi^2 - 2Rp / (x * Rj^3) * hxi * hth + hth^2 / (x * Rj)^2
        D = G * H^(n + 2) * g2^div(n - 1, 2)
        hup = real(Rdot[j]) > 0 ? hh[f+1, j] : hh[f, j]
        FX[f+1, j] = -D * (x * (1 + (Rp / Rj)^2) * hxi - Rp / Rj * hth) - x^2 * Rj * Rdot[j] * hup
    end
    @inbounds for j in 1:Nt, i in 1:Nx
        jp = mod1(j + 1, Nt); x = xi[i]
        Rf = (Rv[j] + Rv[jp]) / 2; Rp = (Rv[jp] - Rv[j]) / ht
        H = (hh[i, j] + hh[i, jp]) / 2
        hth = (hh[i, jp] - hh[i, j]) / ht
        hxi = (hg(i + 1, j) - hg(i - 1, j) + hg(i + 1, jp) - hg(i - 1, jp)) / (4hx)
        g2 = (1 + (Rp / Rf)^2) / Rf^2 * hxi^2 - 2Rp / (x * Rf^3) * hxi * hth + hth^2 / (x * Rf)^2
        D = G * H^(n + 2) * g2^div(n - 1, 2)
        FT[i, j] = -D * (-(Rp / Rf) * hxi + hth / x)
    end
    li = LinearIndices((Nx, Nt))
    @inbounds for j in 1:Nt, i in 1:Nx
        Rj = Rv[j]; x = xi[i]
        div_a = (FX[i+1, j] - FX[i, j]) / hx + (FT[i, j] - FT[i, mod1(j - 1, Nt)]) / ht - x * Rj^2 * amb(m, x * Rj, j)
        r[li[i, j]] = steady ? div_a : x * Rj^2 * hh[i, j] - x * Rold[j]^2 * hold[i, j] + dt * div_a
    end
    @inbounds for j in 1:Nt
        Rj = Rv[j]; Rp = Rp_c[j]
        vxi = (-3vfun(hh[Nx, j]) + vfun(hh[Nx-1, j]) / 3) / hx
        gx = sqrt(1 + (Rp / Rj)^2) / Rj
        r[Nx*Nt+j] = steady ? vxi : 1e3 * hx * Rj * (Rj - Rold[j] + dt * cv * (vxi^2)^div(n - 1, 2) * vxi * gx^(n + 1) * Rj)
    end
    return r
end

# Sparsity pattern of the colour-grouped complex-step Jacobian (same colouring as the original:
# h by (i mod 3, j mod 4), R by j mod 4). Returns (Ii, Jj, groups); group g = (perturbed unknowns,
# rows to read for each pattern entry in order).
struct JacPattern
    Ii::Vector{Int}
    Jj::Vector{Int}
    groups::Vector{Tuple{Vector{Int},UnitRange{Int}}}   # (perturbed unknowns, range of pattern entries)
end

function jac_pattern(m::LandauModel)
    Nx, Nt = m.Nx, m.Nt
    Nt % 4 == 0 || error("Nt must be a multiple of 4")
    li = LinearIndices((Nx, Nt))
    Ii = Int[]; Jj = Int[]; groups = Tuple{Vector{Int},UnitRange{Int}}[]
    for ci in 0:2, cj in 0:3
        k0 = length(Ii) + 1; pert = Int[]
        for j in cj+1:4:Nt, i in ci+1:3:Nx
            col = li[i, j]; push!(pert, col)
            for dj in -1:1, di in -1:1
                r_ = i + di; s_ = mod1(j + dj, Nt)
                1 <= r_ <= Nx && (push!(Ii, li[r_, s_]); push!(Jj, col))
            end
            i >= Nx - 1 && (push!(Ii, Nx * Nt + j); push!(Jj, col))
        end
        push!(groups, (pert, k0:length(Ii)))
    end
    for cj in 0:3
        k0 = length(Ii) + 1; js = collect(cj+1:4:Nt)
        for j in js, dj in -1:1
            s_ = mod1(j + dj, Nt)
            for i in 1:Nx
                push!(Ii, li[i, s_]); push!(Jj, Nx * Nt + j)
            end
            push!(Ii, Nx * Nt + s_); push!(Jj, Nx * Nt + j)
        end
        push!(groups, (Nx * Nt .+ js, k0:length(Ii)))
    end
    return JacPattern(Ii, Jj, groups)
end

mutable struct Solver
    m::LandauModel
    pat::JacPattern
    J::SparseMatrixCSC{Float64,Int}
    nzperm::Vector{Int}        # pattern entry k -> position in J.nzval
    bufR::ResBuf{Float64}
    bufC::ResBuf{ComplexF64}
    zc::Vector{ComplexF64}
    ztr::Vector{Float64}
    F::Any                     # cached UMFPACK factorisation (symbolic analysis reused)
    nnewton::Int
end
function Solver(m::LandauModel)
    pat = jac_pattern(m); K = length(pat.Ii)
    J = sparse(pat.Ii, pat.Jj, Float64.(1:K), m.nunk, m.nunk)
    nnz(J) == K || error("duplicate Jacobian pattern entries")
    nzperm = zeros(Int, K)
    for p in 1:K
        nzperm[Int(J.nzval[p])] = p
    end
    return Solver(m, pat, J, nzperm, ResBuf{Float64}(m), ResBuf{ComplexF64}(m), zeros(ComplexF64, m.nunk), zeros(m.nunk), nothing, 0)
end

# complex-step Jacobian into s.J (fixed pattern, no allocation)
function jacobian!(s::Solver, z, hold, Rold, dt; steady::Bool = false)
    del = 1e-30; pat = s.pat; zc = s.zc; nzv = s.J.nzval
    for (pert, rng) in pat.groups
        zc .= z
        @inbounds for c in pert
            zc[c] = complex(z[c], del)
        end
        dr = residual!(s.bufC, s.m, zc, hold, Rold, dt; steady)
        @inbounds for k in rng
            nzv[s.nzperm[k]] = imag(dr[pat.Ii[k]]) / del
        end
    end
    return s.J
end

function jacobian(m::LandauModel, pat::JacPattern, z, hold, Rold, dt)
    s = Solver(m)
    return copy(jacobian!(s, z, hold, Rold, dt))
end

function factor!(s::Solver, J)
    if s.F === nothing
        s.F = lu(J)
    else
        try
            lu!(s.F, J)
        catch
            s.F = lu(J)
        end
    end
    return s.F
end

# One backward-Euler step, Newton with backtracking line search (identical logic to be_step of the original)
function be_step(s::Solver, h, R, dt)
    m = s.m; Nx, Nt = m.Nx, m.Nt
    z = [vec(h); R]; ch = Inf
    for it in 1:m.max_iters
        s.nnewton += 1
        r = residual!(s.bufR, m, z, h, R, dt); nr = norm(r)
        rhs = -r
        d = factor!(s, jacobian!(s, z, h, R, dt)) \ rhs
        lam = 1.0; ok = false
        for _ in 1:10
            s.ztr .= z .+ lam .* d
            ok = norm(residual!(s.bufR, m, s.ztr, h, R, dt)) <= (1 - 1e-4lam) * nr; ok && break
            lam /= 2
        end
        nh = Nx * Nt
        ch = norm(view(d, 1:nh)) / norm(view(z, 1:nh)) + norm(view(d, nh+1:length(d))) / norm(view(z, nh+1:length(z)))
        if !ok
            ch < m.noise_tol && (z .+= d; return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], true, it)
            return h, R, false, it
        end
        z .+= lam .* d
        all(isfinite, z) || return h, R, false, it
        ch < m.tol && return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], true, it
    end
    return reshape(z[1:Nx*Nt], Nx, Nt), z[Nx*Nt+1:end], ch < 1e-6, m.max_iters
end

# Plain Newton on the steady equations (affine invariant, so row scaling is irrelevant), from a state close to
# steady. Converged when the relative correction < tol; returns (h, R, ok, iterations).
function steady_newton(s::Solver, h, R; tol = 1e-11, max_iters = 25)
    m = s.m; nh = m.Nx * m.Nt
    z = [vec(h); R]
    for it in 1:max_iters
        s.nnewton += 1
        r = residual!(s.bufR, m, z, h, R, 1.0; steady = true)
        d = factor!(s, jacobian!(s, z, h, R, 1.0; steady = true)) \ (-r)
        ch = norm(view(d, 1:nh)) / norm(view(z, 1:nh)) + norm(view(d, nh+1:length(d))) / norm(view(z, nh+1:length(z)))
        (isfinite(ch) && ch < 1e-2) || return h, R, false, it
        z .+= d
        ch < tol && return reshape(z[1:nh], m.Nx, m.Nt), z[nh+1:end], true, it
    end
    return h, R, false, max_iters
end

initial_state(m::LandauModel, par::ModelParams) =
    ([par.H_init * sqrt(max(1 - x^2, 0.0)) for x in m.xi, j in 1:m.Nt], fill(par.R_init, m.Nt))

volume(m::LandauModel, h, R) = sum(m.xi[i] * R[j]^2 * h[i, j] for i in 1:m.Nx, j in 1:m.Nt) * m.hx * m.ht

# Replica of the original script's time loop (adaptive dt, growth 1.5 after easy steps, steady stop on the
# margin rate); used to check that this copy reproduces dense_landau_2d.jl.
function run_replica(par::ModelParams; nbits = 5, dt0 = 1000.0, t_run = 150000.0, steady_tol = 1e-9, report = 10,
                     eismint = false)
    m = LandauModel(par; nbits, eismint); s = Solver(m)
    h, R = initial_state(m, par); t = 0.0
    t_end = t + t_run; dt = dt0; nst = 0; nit = 0; nrej = 0
    wall = @elapsed while t < t_end - 1e-9
        dtt = min(dt, t_end - t)
        hn, Rn, ok, it = be_step(s, h, R, dtt)
        if !ok
            nrej += 1; dt /= 2; dt < 1e-4 && error("dt collapsed"); continue
        end
        rate = norm(Rn - R) / norm(R) / dtt
        h, R, t = hn, Rn, t + dtt; nst += 1; nit += it
        it <= 6 && (dt = min(dt0, 1.5dt))
        nst % report == 0 && println(@sprintf("t = %9.1f a  dt %7.1f  R in [%.3f, %.3f] km  h(pole) %.2f m  newton %d  |dR/dt|/|R| %.1e",
                                              t, dtt, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1], it, rate))
        steady_tol > 0 && rate < steady_tol && break
    end
    @printf("%d steps (%d rejected), mean Newton %.1f, %.1f s\n", nst, nrej, nit / max(nst, 1), wall)
    @printf("t = %.0f a: margin %.3f .. %.3f km, h(pole) %.2f m\n", t, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1])
    return h, R, t
end

const QOI_NAMES = ["V", "hpole", "R_E", "R_N", "R_W"]   # per output time
qoi_labels() = vcat(["$(q)_10ka" for q in QOI_NAMES], ["$(q)_ss" for q in QOI_NAMES])

# QoIs of a state: volume (1e6 km^3), pole thickness (km; theta-mean of the first xi cell), margin radius
# (km) towards theta = 0 (E), pi/2 (N), pi (W). Fixed directions instead of min/max over theta keep the QoIs
# smooth in the parameters (the argmin/argmax direction can jump between grid angles).
function qois(m::LandauModel, h, R)
    Nt = m.Nt
    return [volume(m, h, R) / 1e15, sum(h[1, :]) / Nt / 1e3, R[1] / 1e3, R[Nt÷4+1] / 1e3, R[Nt÷2+1] / 1e3]
end

# Advance by dt with backward Euler, splitting a failing step into two halves recursively.
function substep(s::Solver, h, R, dt, depth, stats)
    hn, Rn, ok, it = be_step(s, h, R, dt)
    ok && return hn, Rn, true
    depth >= 8 && return h, R, false
    stats[:nsplit] += 1
    h1, R1, ok1 = substep(s, h, R, dt / 2, depth + 1, stats)
    ok1 || return h, R, false
    return substep(s, h1, R1, dt / 2, depth + 1, stats)
end

# Fixed transient time grid to t_out: `graded` start dt_tr/16, /16, /8, /4, /2 (sums to dt_tr) then dt_tr steps.
# The same grid is used for every parameter value, so the transient QoIs are smooth functions of the parameters
# (as long as no step has to be split).
function transient_grid(dt_tr, t_out; graded::Bool = true)
    g = graded ? dt_tr .* [1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2] : Float64[]
    nrest = round(Int, (t_out - sum(g)) / dt_tr)
    return vcat(g, fill(dt_tr, nrest))
end

"""
    run_model(par::ModelParams; nbits=5, dt_tr=500, t_out=10000, ss=true, ...) -> (q, info)

Transient from the initial dome (H_init, R_init) to t_out on a fixed time grid (QoIs at t_out), then to
steady state: backward Euler with dt growing x`ss_growth` per easy step; once a step changes (h, R) by less than
`newton_switch` (relative), Newton on the steady equations (margin condition v_xi = 0, quadratic convergence to
the discrete steady state). Fallback if that fails: keep time stepping up to `dt_max` until a step changes the
state by less than `ss_change_tol`. q = [QoIs(t_out); QoIs(steady)] (10 numbers).
"""
function run_model(par::ModelParams; nbits::Int = 5, dt_tr = 500.0, t_out = 10000.0, graded::Bool = false, ss::Bool = true,
                   dt_ss0 = 1000.0, ss_growth = 4.0, dt_max = 1e10, ss_change_tol = 1e-8, max_ss_steps = 200,
                   newton_switch = 1e-4, ss_newton = true, verbose = false)
    m = LandauModel(par; nbits); s = Solver(m)
    h, R = initial_state(m, par)
    stats = Dict(:nsplit => 0, :nrej => 0, :nss => 0)
    t = 0.0; tsplit = Float64[]
    for dtk in transient_grid(dt_tr, t_out; graded)
        ns0 = stats[:nsplit]
        h, R, ok = substep(s, h, R, dtk, 0, stats)
        ok || error("transient step failed at t = $t for $par")
        stats[:nsplit] > ns0 && push!(tsplit, t)
        t += dtk
        verbose && @printf("t = %8.0f  R in [%.3f, %.3f] km  h1 %.2f m  newton %d\n", t, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1], s.nnewton)
    end
    q_tr = qois(m, h, R)
    q_ss = fill(NaN, length(q_tr)); rate = NaN; change = NaN; converged = false; attempts = 0; newton_used = false
    if ss
        dt = dt_ss0
        while stats[:nss] < max_ss_steps
            hn, Rn, ok, it = be_step(s, h, R, dt)
            if !ok
                stats[:nrej] += 1; dt /= 2; dt < 1.0 && error("steady phase: dt collapsed for $par"); continue
            end
            change = max(norm(hn - h) / norm(h), norm(Rn - R) / norm(R)); rate = change / dt
            h, R, t = hn, Rn, t + dt; stats[:nss] += 1
            verbose && @printf("ss: t = %10.3e  dt %9.2e  R in [%.6f, %.6f] km  h1 %.6f m  newton %d  change %.1e\n",
                               t, dt, minimum(R) / 1e3, maximum(R) / 1e3, h[1, 1], it, change)
            if ss_newton && attempts < 3 && change < newton_switch / 100.0^attempts
                attempts += 1
                hs, Rs, oks, its = steady_newton(s, h, R)
                verbose && @printf("ss: steady Newton %s after %d iterations, R in [%.6f, %.6f] km  h1 %.6f m\n", oks ? "converged" : "FAILED",
                                   its, minimum(Rs) / 1e3, maximum(Rs) / 1e3, hs[1, 1])
                if oks
                    h, R = hs, Rs; converged = true; newton_used = true; change = 0.0; break
                end
            end
            if dt >= dt_max && change < ss_change_tol
                converged = true; break
            end
            it <= 6 && (dt = min(dt_max, ss_growth * dt))
        end
        converged || @warn "steady state not reached (change $change at dt $dt_max) for $par"
        q_ss = qois(m, h, R)
    end
    info = (; nnewton = s.nnewton, nsplit = stats[:nsplit], tsplit, nrej = stats[:nrej], nss = stats[:nss], change, converged, newton_used,
            Rmin = minimum(R) / 1e3, Rmax = maximum(R) / 1e3)
    return vcat(q_tr, q_ss), info
end
