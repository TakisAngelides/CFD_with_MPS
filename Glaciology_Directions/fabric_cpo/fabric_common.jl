# Ice crystal-orientation fabric (CPO) as a tensor-train problem: shared physics and tools.
#
# Orientation distribution psi(x, z, n), n on the unit sphere. Lattice rotation of the c-axes
#     dn/dt = W n + iota (D n - (n.D.n) n) = K n - (n.K n) n,   K = W + iota D,
# (iota = -1: c-axes rotate like plane normals, i.e. towards the compression axis; iota = +1: like
# material lines). The solution is n(t) = G n0 / |G n0| with dG/dt = K G (along the particle path),
# det G = 1 (tr K = 0), and the push-forward of an isotropic initial distribution is the angular
# central Gaussian
#     psi(n) = 1 / (4 pi) |G^{-1} n|^{-3}                      (any iota, any path history).
#
# Plane flow in (x, z) with y the neutral axis: the orientation grid uses the y axis as polar axis,
#     n = (sqrt(1 - mu^2) cos(phi), mu, sqrt(1 - mu^2) sin(phi)),   mu in (-1, 1), phi in [0, 2pi)
# (equal-area cells, unit Jacobian dA = dmu dphi; e_y is a fixed point of the lattice rotation in
# plane flow, so the polar singularity of the coordinates sits where nothing moves).

using LinearAlgebra
using Printf
using FFTW

# ------------------------------------------------------------------------------------------------
# Flow: divide-to-flank plane flow (dimensionless: ice thickness H = 1, accumulation a = 1).
# Stream function -x omega(zeta): u = x omega'(zeta), w = -omega(zeta), incompressible.
# omega = wb + (1 - wb) (fs zeta + (1 - fs) Omega_SIA(zeta)) with Omega_SIA the Glen-n SIA shape
# (no slip at the bed), wb the basal melt fraction (finite age and strain at the bed), fs a sliding
# fraction (plug flow). Particle paths: x omega(zeta) = const, so the parcel at (x, zeta) entered at
# the surface at x_s = x omega(zeta).
# ------------------------------------------------------------------------------------------------

struct DivideFlow
    n::Float64
    wb::Float64
    fs::Float64
end
DivideFlow(; n = 3.0, wb = 0.2, fs = 0.0) = DivideFlow(n, wb, fs)

Om(z, n) = (n + 2) / (n + 1) * (z + ((1 - z)^(n + 2) - 1) / (n + 2))
dOm(z, n) = (n + 2) / (n + 1) * (1 - (1 - z)^(n + 1))
d2Om(z, n) = (n + 2) * (1 - z)^n

omega(f::DivideFlow, z) = f.wb + (1 - f.wb) * (f.fs * z + (1 - f.fs) * Om(z, f.n))
domega(f::DivideFlow, z) = (1 - f.wb) * (f.fs + (1 - f.fs) * dOm(z, f.n))
d2omega(f::DivideFlow, z) = (1 - f.wb) * (1 - f.fs) * d2Om(z, f.n)

# velocity gradient L_ij = du_i/dx_j in the (x, z) plane as (Lxx, Lxz, Lzx, Lzz)
velgrad(f::DivideFlow, x, z) = (domega(f, z), x * d2omega(f, z), 0.0, -domega(f, z))

# K = W + iota D (plane block), as (Kxx, Kxz, Kzx, Kzz)
function Kmat(L, iota)
    Lxx, Lxz, Lzx, Lzz = L
    D = (Lxz + Lzx) / 2
    W = (Lxz - Lzx) / 2                       # W_xz = -W_zx
    return (iota * Lxx, W + iota * D, -W + iota * D, iota * Lzz)
end

# The three scalar fields of K used by the orientation operator:
#   e = Kxx = -Kzz, d = (Kxz + Kzx) / 2, w = (Kzx - Kxz) / 2
Kedw(K) = (K[1], (K[2] + K[3]) / 2, (K[3] - K[2]) / 2)

mul2(A, B) = (A[1] * B[1] + A[2] * B[3], A[1] * B[2] + A[2] * B[4],
              A[3] * B[1] + A[4] * B[3], A[3] * B[2] + A[4] * B[4])
add2(A, B, s) = (A[1] + s * B[1], A[2] + s * B[2], A[3] + s * B[3], A[4] + s * B[4])

# G at (x, zeta) for the steady divide flow: integrate dG/dzeta' = -K(x(zeta'), zeta') G / omega(zeta')
# from the surface zeta' = 1 down to zeta (RK4, nsteps uniform steps).
function path_G(f::DivideFlow, x, z; iota = -1.0, nsteps = 400)
    G = (1.0, 0.0, 0.0, 1.0)
    z >= 1 && return G
    c = x * omega(f, z)                       # streamline constant
    rhs(zz, G) = begin
        K = Kmat(velgrad(f, c / omega(f, zz), zz), iota)
        s = -1 / omega(f, zz)
        mul2((s * K[1], s * K[2], s * K[3], s * K[4]), G)
    end
    h = (z - 1) / nsteps
    zz = 1.0
    for _ in 1:nsteps
        k1 = rhs(zz, G)
        k2 = rhs(zz + h / 2, add2(G, k1, h / 2))
        k3 = rhs(zz + h / 2, add2(G, k2, h / 2))
        k4 = rhs(zz + h, add2(G, k3, h))
        G = (G[1] + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1]), G[2] + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2]),
             G[3] + h / 6 * (k1[3] + 2k2[3] + 2k3[3] + k4[3]), G[4] + h / 6 * (k1[4] + 2k2[4] + 2k3[4] + k4[4]))
        zz += h
    end
    return G
end

# G = exp(t K) for a constant K (local deformation without advection), via 2x2 closed form:
# K traceless, K^2 = -det(K) I, so exp(tK) = cosh(s t) I + sinh(s t)/s K with s^2 = -det K.
function expK(K, t)
    s2 = -(K[1] * K[4] - K[2] * K[3])
    if abs(s2) * t^2 < 1e-12
        a, b = 1.0 + s2 * t^2 / 2, t * (1 + s2 * t^2 / 6)
    elseif s2 > 0
        s = sqrt(s2); a, b = cosh(s * t), sinh(s * t) / s
    else
        s = sqrt(-s2); a, b = cos(s * t), sin(s * t) / s
    end
    return (a + b * K[1], b * K[2], b * K[3], a + b * K[4])
end

inv2(G) = (dt = G[1] * G[4] - G[2] * G[3]; (G[4] / dt, -G[2] / dt, -G[3] / dt, G[1] / dt))

# M = B^T B with B = G^{-1}; psi(n) = 1/(4pi) (n_xz . M n_xz + n_y^2)^{-3/2}. Returned as (Mxx, Mxz, Mzz).
function Mfrom(G)
    B = inv2(G)
    return (B[1]^2 + B[3]^2, B[1] * B[2] + B[3] * B[4], B[2]^2 + B[4]^2)
end

@inline function psi_closed(M, mu, phi)
    r2 = 1 - mu^2
    c, s = cos(phi), sin(phi)
    q = r2 * (M[1] * c^2 + 2 * M[2] * c * s + M[3] * s^2) + mu^2
    return q^(-1.5) / (4pi)
end

# general (non plane-strain) version: M3 = (Mxx, Mxy, Mxz, Myy, Myz, Mzz) of M = B^T B, B = G^{-1}
@inline function psi_closed3(M, mu, phi)
    r = sqrt(1 - mu^2)
    nx, ny, nz = r * cos(phi), mu, r * sin(phi)
    q = M[1] * nx^2 + M[4] * ny^2 + M[6] * nz^2 + 2 * (M[2] * nx * ny + M[3] * nx * nz + M[5] * ny * nz)
    return q^(-1.5) / (4pi)
end

# smallest / largest eigenvalue of M (plane strain: eigenvalues of the 2x2 block and 1). The ODF peak
# value is lam_min^{-3/2}/(4pi) and its angular half-width towards the stiffest direction ~ sqrt(lam_min/lam_max).
function Meigs(M)
    tr = M[1] + M[3]; dt = M[1] * M[3] - M[2]^2
    disc = sqrt(max(tr^2 / 4 - dt, 0.0))
    l1, l2 = tr / 2 - disc, tr / 2 + disc
    return min(l1, 1.0), max(l2, 1.0)
end

# ------------------------------------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------------------------------------
cellc(lo, hi, N) = [lo + (i - 0.5) * (hi - lo) / N for i in 1:N]
mu_grid(N) = cellc(-1.0, 1.0, N)
phi_grid(N) = cellc(0.0, 2pi, N)

# Dense psi on space points (list of M triples) x (mu, phi) grid -> array (Nspace..., Nmu, Nphi)
function dense_psi(Ms::AbstractArray, mus, phis)
    A = zeros(size(Ms)..., length(mus), length(phis))
    cs = CartesianIndices(Ms)
    Threads.@threads for j in eachindex(phis)
        for i in eachindex(mus), I in cs
            A[I, i, j] = psi_closed(Ms[I], mus[i], phis[j])
        end
    end
    return A
end

# ------------------------------------------------------------------------------------------------
# Gauss-Legendre (Golub-Welsch) and fully normalised associated Legendre functions
# Pbar[l+1, m+1, k] = Pbar_lm(x_k) with int_{-1}^{1} Pbar_lm^2 dx = 1.
# Real SH: Y_l0 = Pbar_l0 / sqrt(2pi), Y_lm^{c,s} = Pbar_lm {cos, sin}(m phi) / sqrt(pi).
# ------------------------------------------------------------------------------------------------
function gauss_legendre(N)
    b = [k / sqrt(4k^2 - 1) for k in 1:N-1]
    E = eigen(SymTridiagonal(zeros(N), b))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

function legendre_table(lmax, x)
    P = zeros(length(x), lmax + 1, lmax + 1)          # P[k, l+1, m+1]
    for (k, xk) in enumerate(x)
        s = sqrt(1 - xk^2)
        pmm = sqrt(0.5)
        for m in 0:lmax
            m > 0 && (pmm *= sqrt((2m + 1) / (2m)) * s)
            P[k, m+1, m+1] = pmm
            m == lmax && break
            P[k, m+2, m+1] = sqrt(2m + 3) * xk * pmm
            for l in m+2:lmax
                a = sqrt((4l^2 - 1) / (l^2 - m^2))
                b = sqrt(((l - 1)^2 - m^2) / (4 * (l - 1)^2 - 1))
                P[k, l+1, m+1] = a * (xk * P[k, l, m+1] - b * P[k, l-1, m+1])
            end
        end
    end
    return P
end

# Degree-energy spectrum E_l = sum_m c_lm^2 of a function given on the GL x uniform-phi quadrature grid
# F[k, j] = f(x_k, phi_j), via an FFT in phi (phi_j cell centred; the phase drops out of |c_lm|^2).
struct SHProjector
    lmax::Int
    w::Vector{Float64}
    P::Array{Float64,3}
    hphi::Float64
end

function make_sh(lmax, Ngl, Nphi)
    x, w = gauss_legendre(Ngl)
    return SHProjector(lmax, w, legendre_table(lmax, x), 2pi / Nphi), x, phi_grid(Nphi)
end

function sh_energy(S::SHProjector, F::AbstractMatrix, plan)
    Fh = (plan * F) .* S.hphi
    Fw = Fh .* S.w
    E = zeros(S.lmax + 1)
    for m in 0:min(S.lmax, size(Fh, 2) - 1)
        nm = m == 0 ? 2pi : pi
        f = @view Fw[:, m+1]
        for l in m:S.lmax
            c = zero(ComplexF64)
            P = @view S.P[:, l+1, m+1]
            @inbounds @simd for k in eachindex(f)
                c += P[k] * f[k]
            end
            E[l+1] += abs2(c) / nm
        end
    end
    return E
end

# Energy spectra of many functions in parallel (one FFTW plan per thread, static scheduling)
function sh_energies(S::SHProjector, Ffun, npts, Ngl, Nphi)
    Es = zeros(S.lmax + 1, npts)
    plans = [plan_rfft(zeros(Ngl, Nphi), 2) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for p in 1:npts
        Es[:, p] = sh_energy(S, Ffun(p), plans[Threads.threadid()])
    end
    return Es
end

# Second-order orientation tensor eigenvalues from a function on the GL x phi quadrature grid
function a2_eigs(F, x, w, phis)
    A = zeros(3, 3); hphi = 2pi / length(phis)
    for (j, p) in enumerate(phis), (k, mu) in enumerate(x)
        r = sqrt(1 - mu^2); n = (r * cos(p), mu, r * sin(p))
        wt = w[k] * hphi * F[k, j]
        for a in 1:3, b in 1:3
            A[a, b] += wt * n[a] * n[b]
        end
    end
    return sort(eigvals(Symmetric(A)), rev = true), tr(A)
end

# ------------------------------------------------------------------------------------------------
# TT-SVD of a dense tensor with a global relative Frobenius tolerance eps (per-bond eps/sqrt(d-1)).
# Returns the bond ranks, the number of parameters and (optionally) the cores.
# ------------------------------------------------------------------------------------------------
function tt_svd(A::AbstractArray, eps; keep_cores = false)
    dims = size(A); d = length(dims)
    nrmA = norm(A)
    delta = eps / sqrt(max(d - 1, 1)) * nrmA
    C = Matrix(reshape(A, dims[1], :))
    r = 1
    ranks = Int[]; cores = Array{Float64,3}[]
    for k in 1:d-1
        C = reshape(C, r * dims[k], :)
        F = svd(C)
        s = F.S
        tail2 = reverse(cumsum(reverse(s .^ 2)))   # tail2[i] = sum(s[i:end].^2)
        rnew = length(s)
        for i in 1:length(s)
            t = i == length(s) ? 0.0 : tail2[i+1]
            if t <= delta^2
                rnew = i
                break
            end
        end
        keep_cores && push!(cores, reshape(F.U[:, 1:rnew], r, dims[k], rnew))
        C = Diagonal(s[1:rnew]) * F.Vt[1:rnew, :]
        push!(ranks, rnew)
        r = rnew
    end
    keep_cores && push!(cores, reshape(C, r, dims[d], 1))
    rr = [1; ranks; 1]
    npar = sum(rr[k] * dims[k] * rr[k+1] for k in 1:d)
    return ranks, npar, cores
end

function tt_full(cores)
    T = reshape(cores[1], size(cores[1], 2), :)
    for k in 2:length(cores)
        c = cores[k]
        T = reshape(T * reshape(c, size(c, 1), :), :, size(c, 3))
    end
    return vec(T)
end

# ------------------------------------------------------------------------------------------------
# Quantics orderings. A dense array with dims (2^n1, 2^n2, ...) (column major) is reshaped into
# binary dims [x1_lsb .. x1_msb, x2_lsb .. ]; an ordering is a list of (dim, k) pairs (k = 1 MSB).
# ------------------------------------------------------------------------------------------------
function quantics_perm(nbits::Vector{Int}, order::Vector{Tuple{Int,Int}})
    off = cumsum([0; nbits[1:end-1]])
    return [off[a] + (nbits[a] - k + 1) for (a, k) in order]
end

seq_order(dims, nbits) = [(a, k) for a in dims for k in 1:nbits[a]]
function inter_order(dims, nbits)
    o = Tuple{Int,Int}[]
    for k in 1:maximum(nbits[a] for a in dims), a in dims
        k <= nbits[a] && push!(o, (a, k))
    end
    return o
end

function quantics_tensor(A::AbstractArray, nbits, order)
    B = reshape(A, fill(2, sum(nbits))...)
    return permutedims(B, quantics_perm(nbits, order))
end

parse_opts(defaults) = begin
    o = copy(defaults)
    for a in ARGS
        k, v = split(a, "=", limit = 2); o[k] = v
    end
    o
end

# SH degree energies of one function on a GL x phi grid without a Legendre table (large lmax)
function sh_energy_stream(F::AbstractMatrix, x, w, hphi, lmax)
    Fh = rfft(F, 2) .* hphi .* w
    E = zeros(lmax + 1)
    s = sqrt.(1 .- x .^ 2)
    pmm = fill(sqrt(0.5), length(x))
    p0 = similar(pmm); p1 = similar(pmm); p2 = similar(pmm)
    for m in 0:min(lmax, size(Fh, 2) - 1)
        if m > 0
            c = sqrt((2m + 1) / (2m))
            @inbounds for k in eachindex(pmm)
                v = pmm[k] * c * s[k]
                pmm[k] = abs(v) < 1e-280 ? 0.0 : v
            end
        end
        nm = m == 0 ? 2pi : pi
        f = @view Fh[:, m+1]
        p1 .= pmm
        E[m+1] += abs2(sum(p1 .* f)) / nm
        m == lmax && break
        p2 .= sqrt(2m + 3) .* x .* pmm
        E[m+2] += abs2(sum(p2 .* f)) / nm
        for l in m+2:lmax
            a = sqrt((4l^2 - 1) / (l^2 - m^2)); b = sqrt(((l - 1)^2 - m^2) / (4 * (l - 1)^2 - 1))
            p0 .= p1; p1 .= p2
            cl = zero(ComplexF64)
            @inbounds @simd for k in eachindex(p2)
                p2[k] = a * (x[k] * p1[k] - b * p0[k])
                cl += p2[k] * f[k]
            end
            E[l+1] += abs2(cl) / nm
        end
    end
    return E
end
