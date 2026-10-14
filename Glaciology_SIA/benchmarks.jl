# ---------------------------------------------------------------------------------------------
# Standard isothermal SIA benchmarks (lengths in m, time in years)
#
# 1) Halfar similarity solution = Bueler et al. (2005) Test B in 2D (exact, time dependent, a = 0)
#    Bueler, Lingle, Kallen-Brown, Covey, Bowman (2005), J. Glaciol. 51(173), 291-306.
#    Generalised to d = 1 (flowline) and d = 2 (radial):
#       beta = 1 / ((2n+1) d + n + 1),  alpha = d beta
#       H(t, r) = H0 (t0/t)^alpha [1 - ((t0/t)^beta r/R0)^((n+1)/n)]_+^(n/(2n+1))
#       t0 = (beta/Gamma) ((2n+1)/(n+1))^n R0^(n+1) / H0^(2n+1)
#    Test B: n = 3, A = 1e-16 Pa^-3 a^-1, rho = 910, g = 9.81, H0 = 3600 m, R0 = 750 km.
#
# 2) EISMINT I moving-margin experiment (Huybrechts et al., 1996, Ann. Glaciol. 23)
#    Flat bed, 1500 km x 1500 km, a = min(Mmax, S (R_el - r)), Mmax = 0.5 m/a, S = 1e-5 a^-1,
#    R_el = 450 km, start from no ice, integrate to steady state (~ 50-200 ka).
#    No exact solution for H, but the steady margin follows from int a dA = 0 (eismint_margin).
# ---------------------------------------------------------------------------------------------

Base.@kwdef struct Halfar
    H0::Float64 = 3600.0
    R0::Float64 = 750e3
    d::Int = 2
    p::SIAParams = SIAParams()
end

halfar_beta(h::Halfar) = 1 / ((2h.p.n_glen + 1) * h.d + h.p.n_glen + 1)
halfar_alpha(h::Halfar) = h.d * halfar_beta(h)

function halfar_t0(h::Halfar)
    n = h.p.n_glen
    return (halfar_beta(h) / Gamma(h.p)) * ((2n + 1) / (n + 1))^n * h.R0^(n + 1) / h.H0^(2n + 1)
end

function halfar_H(h::Halfar, t, r)
    n = h.p.n_glen
    t0 = halfar_t0(h)
    tau = t0 / t
    inner = 1 - (tau^halfar_beta(h) * r / h.R0)^((n + 1) / n)
    return inner > 0 ? h.H0 * tau^halfar_alpha(h) * inner^(n / (2n + 1)) : 0.0
end

halfar_margin(h::Halfar, t) = h.R0 * (t / halfar_t0(h))^halfar_beta(h)

# Volume of the exact solution, d = 1: per unit width, d = 2: total
function halfar_volume(h::Halfar, t)
    n = h.p.n_glen
    t0 = halfar_t0(h)
    k = n / (2n + 1); m = (n + 1) / n
    # int_0^1 (1 - x^m)^k x^(d-1) dx = B(d/m, k+1)/m
    beta_fn(a, b) = exp(lgamma_(a) + lgamma_(b) - lgamma_(a + b))
    I = beta_fn(h.d / m, k + 1) / m
    Hc = h.H0 * (t0 / t)^halfar_alpha(h)
    R = halfar_margin(h, t)
    return h.d == 1 ? 2 * Hc * R * I : 2pi * Hc * R^2 * I
end

# log-gamma without extra dependencies (Lanczos, plenty for diagnostics)
function lgamma_(x)
    g = 7
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
         -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
         1.5056327351493116e-7)
    x < 0.5 && return log(pi / abs(sin(pi * x))) - lgamma_(1 - x)
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 1:g+1
        a += c[i+1] / (x + i)
    end
    return 0.5 * log(2pi) + (x + 0.5) * log(t) - t + log(a)
end

Base.@kwdef struct EISMINTMovingMargin
    L::Float64 = 750e3        # half width, domain [-L, L]^d
    Mmax::Float64 = 0.5       # m/a
    S::Float64 = 1e-5         # a^-1 (1e-2 m/a per km)
    R_el::Float64 = 450e3
    p::SIAParams = SIAParams()
end

eismint_mb(e::EISMINTMovingMargin, r) = min(e.Mmax, e.S * (e.R_el - r))

# Steady-state margin R: int_0^R a(r) r dr = 0 (2D), int_0^R a(r) dr = 0 (1D)
function eismint_margin(e::EISMINTMovingMargin, d)
    F(R) = begin
        rs = range(0, R, length = 20001)
        w = d == 1 ? ones(length(rs)) : collect(rs)
        sum(eismint_mb(e, r) * wi for (r, wi) in zip(rs, w)) * step(rs)
    end
    lo, hi = e.R_el, 2e6
    for _ in 1:80
        mid = (lo + hi) / 2
        F(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
