# isostasy.jl -- physics of the FastIsostasy (LV-)ELVA model in Fourier space, loads, analytic
# solutions, dense FFT reference solver and the MPS (QFT) pipeline.
#
# Homogeneous ELVA (elastic plate over a viscous half space, Bueler et al. 2007; FastIsostasy's
# lv_elva! with constant viscosity and rigidity):
#     2 eta |k| d/dt u^ = -(rho_m g + D |k|^4) u^ - sigma^ ,     sigma = rho_ice g H
# For a load that is constant over [t0, t0 + dt] the exact solution is the exponential integrator
#     u^(t0+dt) = E u^(t0) - (1 - E) sigma^ / beta,   E = exp(-lambda dt),
#     beta = rho_m g + D |k|^4,   lambda = beta / (2 eta |k|),
# i.e. two Fourier multipliers and no linear solve. lambda(0) = Inf (k = 0 relaxes instantly).

include("qft.jl")
using FFTW, HDF5

const SEC_PER_YEAR = 60^2 * 24 * 365.25

Base.@kwdef struct Earth
    g::Float64 = 9.8
    rho_ice::Float64 = 910.0
    rho_m::Float64 = 3400.0
    eta::Float64 = 1e21
    E::Float64 = 6.6e10
    nu::Float64 = 0.28
    Tlitho::Float64 = 88e3
end
rigidity(p::Earth) = p.E * p.Tlitho^3 / (12 * (1 - p.nu^2))
beta(k, p::Earth) = p.rho_m * p.g + rigidity(p) * k^4
lambda(k, p::Earth) = k == 0 ? Inf : beta(k, p) / (2 * p.eta * k)
flexural_length(p::Earth) = (rigidity(p) / (p.rho_m * p.g))^0.25

# Multipliers as functions of |k| (t in years). All map sigma^ (Pa) to u^ (m) unless stated.
mult_eq(k, p) = -1 / beta(k, p)                                   # equilibrium (t -> Inf)
mult_resp(k, t, p) = -(1 - exp(-lambda(k, p) * t * SEC_PER_YEAR)) / beta(k, p)   # from rest, constant load
mult_decay(k, dt, p) = exp(-lambda(k, p) * dt * SEC_PER_YEAR)     # u^ -> u^ (dimensionless)
mult_bouss(k; mu = 6.6e10 / (2 * 1.28), nu = 0.28) = k == 0 ? 0.0 : -(1 - nu) / (mu * k)  # elastic half space

# ---------------------------------------------------------------------------------------------
# Domain / grids
# ---------------------------------------------------------------------------------------------

# Square periodic domain [-W, W]^2 with N = 2^n cells per dimension, cell centres
cellcentres(N, W) = [-W + (i + 0.5) * (2W / N) for i in 0:N-1]

function kgrid2(N, L)
    k1 = fftk(N, L)
    return [sqrt(kx^2 + ky^2) for kx in k1, ky in k1]
end

# Multiplier as dense array on the FFT grid (d = 2)
mult_array(f, N, L) = [f(kk) for kk in kgrid2(N, L)]

# Multiplier as MPS on the rev (Fourier) bit order, by TCI. Scaled by `scale` (e.g. 1/N^2).
function mult_mps(f, g::QGrid, L; scale = 1.0, tolerance = 1e-13, maxbonddim = 400, cutoff = 1e-15)
    N = npts(g)
    h(idx) = scale * f(kabs(idx, N, L))
    psi, r, err = tci_mps(h, g; rev = true, T = Float64, tolerance = tolerance, maxbonddim = maxbonddim)
    truncate!(psi; cutoff = cutoff)
    return psi
end

# ---------------------------------------------------------------------------------------------
# Loads (ice thickness in m) as functions of position
# ---------------------------------------------------------------------------------------------

disc_sharp(x, y; R0 = 1000e3, H0 = 1000.0) = (x^2 + y^2 < R0^2) ? H0 : 0.0
disc_smooth(x, y; R0 = 1000e3, H0 = 1000.0, w = 20e3) = H0 * 0.5 * (1 - tanh((sqrt(x^2 + y^2) - R0) / w))

# ---------------------------------------------------------------------------------------------
# Analytic solutions (Hankel transforms), FastIsostasy analytic_solutions.jl generalised
#   u(r, t) = -rho_ice g int_0^Inf (1 - e^{-lambda t}) Hhat(kappa) / beta J0(kappa r) kappa dkappa
#   Hhat(kappa) = int_0^Inf H(r) J0(kappa r) r dr     (disc: H0 R0 J1(kappa R0) / kappa)
# ---------------------------------------------------------------------------------------------

using SpecialFunctions, QuadGK

hhat_disc(kappa; R0 = 1000e3, H0 = 1000.0) = H0 * R0 * besselj1(kappa * R0) / kappa

# smooth tanh disc: integrate by parts, Hhat = (H0 / (2 w kappa)) int sech^2((r-R0)/w) r J1(kappa r) dr
const _GL = let
    # 512-point Gauss-Legendre on [-1, 1] via Golub-Welsch
    m = 512
    b = [i / sqrt(4i^2 - 1) for i in 1:m-1]
    F = eigen(SymTridiagonal(zeros(m), b))
    (F.values, 2 .* F.vectors[1, :] .^ 2)
end
function hhat_smooth(kappa; R0 = 1000e3, H0 = 1000.0, w = 20e3)
    x, wt = _GL
    a, b = max(0.0, R0 - 20w), R0 + 20w
    s = 0.0
    for (xi, wi) in zip(x, wt)
        r = (a + b) / 2 + (b - a) / 2 * xi
        s += wi * sech((r - R0) / w)^2 * r * besselj1(kappa * r)
    end
    return H0 / (2w * kappa) * s * (b - a) / 2
end

function analytic_u(r, t, p::Earth; hhat = hhat_disc, rtol = 1e-9)
    f(kappa) = begin
        tf = isinf(t) ? 1.0 : (1 - exp(-lambda(kappa, p) * t * SEC_PER_YEAR))
        -p.rho_ice * p.g * tf * hhat(kappa) / beta(kappa, p) * besselj0(kappa * r) * kappa
    end
    knots = [0.0; 10 .^ (-9:0.02:-3)]
    return sum(quadgk(f, knots[i], knots[i+1]; rtol = rtol)[1] for i in 1:length(knots)-1)
end

# ---------------------------------------------------------------------------------------------
# Dense FFT reference: u = Re ifft( m .* fft(sigma) ), then corner BC (FastIsostasy apply_bc!)
# ---------------------------------------------------------------------------------------------

function fft_solve(sigma::AbstractMatrix, M::AbstractMatrix; plan = nothing, iplan = nothing)
    sh = plan === nothing ? fft(complex.(sigma)) : plan * complex.(sigma)
    sh .*= M
    u = real.(iplan === nothing ? ifft(sh) : iplan * sh)
    return u
end
corner_bc!(u::AbstractMatrix) = (u .-= (u[1, 1] + u[1, end] + u[end, 1] + u[end, end]) / 4; u)

# ---------------------------------------------------------------------------------------------
# MPS pipeline: u = Re( Q^+ ( m .* (Q sigma) ) ), multiplier already scaled by 1/N^d
# ---------------------------------------------------------------------------------------------

realpart(psi::MPS; cutoff = 1e-14, maxdim = typemax(Int)) = lincomb([0.5, 0.5], [psi, dag(psi)]; cutoff = cutoff, maxdim = maxdim)

struct Pipeline
    g::QGrid
    Q::MPO
    Qa::MPO
end
Pipeline(g::QGrid; qcut = 1e-13) = (Q = qft_mpo(g; cutoff = qcut); Pipeline(g, Q, adjoint_mpo(Q)))

function mps_solve(P::Pipeline, sigma::MPS, m::MPS; cutoff = 1e-12, alg = "naive", maxdim = typemax(Int), info = nothing)
    t0 = time()
    sh = apply(P.Q, sigma; alg = alg, cutoff = cutoff, maxdim = maxdim)
    t1 = time()
    uh = apply(diag_mpo(m), sh; alg = "naive", cutoff = cutoff, maxdim = maxdim)
    t2 = time()
    u = apply(P.Qa, uh; alg = alg, cutoff = cutoff, maxdim = maxdim)
    t3 = time()
    u = realpart(u; cutoff = cutoff, maxdim = maxdim)
    t4 = time()
    if info !== nothing
        info[:chi_sigmahat] = maxlinkdim(sh); info[:chi_uhat] = maxlinkdim(uh); info[:chi_u] = maxlinkdim(u)
        info[:t_qft] = t1 - t0; info[:t_mult] = t2 - t1; info[:t_iqft] = t3 - t2; info[:t_real] = t4 - t3
    end
    return u
end

function corner_bc(u::MPS, g::QGrid; cutoff = 1e-14)
    E = MPSEval(u)
    N = npts(g)
    c = sum(real(evaluate(E, g, idx)) for idx in ((0, 0), (0, N - 1), (N - 1, 0), (N - 1, N - 1))) / 4
    return lincomb([1.0, -c], [u, constant_mps(g, 1.0)]; cutoff = cutoff)
end

# values of an MPS along the row j (0-based y index), at x indices is
row_values(u::MPS, g::QGrid, is, j) = (E = MPSEval(u); [real(evaluate(E, g, (i, j))) for i in is])

# ---------------------------------------------------------------------------------------------
# Antarctic load: BedMachine ice thickness on the yelmo ANT-8KM grid (761 x 761, 8 km), grounded
# ice only (flotation criterion), zero padded to 1024 x 1024 (8192 km periodic domain).
# Finer grids (2^n, n > 10) are piecewise-constant upsamplings: in MPS form this appends rank-1
# bits, so the load rank is unchanged.
# ---------------------------------------------------------------------------------------------

const ANT_FILE = "/albedo/work/projects/p_forclima/ice_data/Antarctica/ANT-8KM/ANT-8KM_TOPO-BedMachine.nc"

function antarctic_load(; Npad = 1024, rho_ice = 910.0, rho_sw = 1028.0)
    f = h5open(ANT_FILE, "r")
    H = Float64.(read(f["H_ice"])); zb = Float64.(read(f["z_bed"]))
    close(f)
    grounded = (H .> 0) .& ((zb .>= 0) .| (H .* rho_ice ./ rho_sw .> -zb))
    Hg = H .* grounded
    n0 = size(Hg, 1)
    off = (Npad - n0) ÷ 2
    A = zeros(Npad, Npad)
    A[off+1:off+n0, off+1:off+n0] .= Hg
    return A, 8e3 * Npad / 2     # thickness (m) on the padded grid and half width W (m)
end

# piecewise-constant upsampling of a dense field by 2^m per dimension
upsample(A::AbstractMatrix, m::Int) = m == 0 ? A : repeat(A, inner = (2^m, 2^m))

# MPS version for the interleaved ordering: fine bits are the trailing sites, appended as rank-1 |+>
function upsample_mps(psi::MPS, g::QGrid, gf::QGrid)
    (g.ordering == :interleaved && gf.ordering == :interleaved && gf.n >= g.n) || error("interleaved only")
    ts = ITensor[]
    N = nsites(g)
    for j in 1:N
        push!(ts, replaceind(psi[j], siteind(psi, j), gf.sites[j]))
    end
    lk = Index(1, "Link,up")
    ts[end] = ts[end] * onehot(lk => 1)
    for j in N+1:nsites(gf)
        s = gf.sites[j]
        if j < nsites(gf)
            l2 = Index(1, "Link,up$j")
            push!(ts, ITensor(ones(2), s) * onehot(lk => 1) * onehot(l2 => 1))
            lk = l2
        else
            push!(ts, ITensor(ones(2), s) * onehot(lk => 1))
        end
    end
    return MPS(ts)
end

# vertex grid with the origin on a grid point (index N/2); grids nest under refinement (i -> 2i)
gridpoints(N, W) = [-W + i * (2W / N) for i in 0:N-1]

# Analytic profiles on many radii and times with a fixed composite Gauss-Legendre rule in kappa
# (shared nodes, so Hhat is evaluated once per node). Panels: [0,1e-9] and log panels of 0.01 decade
# to 1e-3, 16 nodes each; resolves J0(kappa r) for r <= 3000 km where the integrand is not negligible.
function analytic_profiles(rs, ts, p::Earth; hhat = hhat_disc)
    xg, wg = let m = 16
        b = [i / sqrt(4i^2 - 1) for i in 1:m-1]
        F = eigen(SymTridiagonal(zeros(m), b))
        (F.values, 2 .* F.vectors[1, :] .^ 2)
    end
    edges = [0.0; 10 .^ (-9:0.01:-3)]
    ks = Float64[]; ws = Float64[]
    for i in 1:length(edges)-1
        a, b = edges[i], edges[i+1]
        append!(ks, (a + b) / 2 .+ (b - a) / 2 .* xg); append!(ws, (b - a) / 2 .* wg)
    end
    Hh = hhat.(ks)
    U = zeros(length(rs), length(ts))
    for (it, t) in enumerate(ts)
        tf = isinf(t) ? ones(length(ks)) : 1 .- exp.(-lambda.(ks, Ref(p)) .* t .* SEC_PER_YEAR)
        c = -p.rho_ice * p.g .* ws .* tf .* Hh ./ beta.(ks, Ref(p)) .* ks
        for (ir, r) in enumerate(rs)
            U[ir, it] = sum(c .* besselj0.(ks .* r))
        end
    end
    return U
end

# Values of a 2D MPS on the coarse lattice (i_c * 2^(n-nc), j_c * 2^(n-nc)), i_c, j_c in 0:2^nc-1:
# the fine bits are projected on 0 and the remaining 2 nc sites contracted exactly.
function coarse_samples(psi::MPS, g::QGrid, nc::Int)
    gc = QGrid(g.d, nc; ordering = g.ordering)
    T = ITensor(1.0)
    keep = Index[]
    bitof = Dict{Int,Tuple{Int,Int}}()
    for a in 1:g.d, k in 1:g.n
        bitof[g.pos[a][k]] = (a, k)
    end
    for j in 1:nsites(g)
        a, k = bitof[j]
        s = siteind(psi, j)
        A = psi[j]
        if k > nc
            A = A * onehot(s => 1)
        else
            push!(keep, s)
        end
        T = T * A
    end
    # keep is in chain order, which is the site order of the coarse grid gc for both orderings
    Tarr = Array(T, keep...)
    A = permutedims(Tarr, invperm(site_perm(gc)))
    return reshape(A, ntuple(_ -> 2^nc, g.d))
end
