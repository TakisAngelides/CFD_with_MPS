# MPS (quantics tensor train) machinery for the fabric equation
#     d psi/dt = sum_k a_k(r) O_k psi + lambda Lap_S psi,   (a_1, a_2, a_3) = (e, d, w)(r)
# on a grid (space bits) x (mu bits) x (phi bits) with an arbitrary bit ordering.
#   O_e = Dmu[g] C[cos 2phi] + Dphi[sin 2phi],  O_d = Dmu[g] C[sin 2phi] - Dphi[cos 2phi],  O_w = -Dphi[1]
# with g(mu) = mu (1 - mu^2), Dmu[g] psi = d_mu(g psi) and Dphi[s] psi = d_phi(s psi) as conservative
# central finite volumes (faces between cells, zero flux through the poles mu = +-1, periodic phi), and
# Lap_S = d_mu((1 - mu^2) d_mu) + (1 - mu^2)^{-1} d_phi^2.
# This is exactly d_t psi = -div_S(psi ndot) for ndot = K n - (n.K n) n in plane flow (see fabric_common.jl).

include(joinpath(@__DIR__, "..", "..", "Glaciology_SIA", "quantics.jl"))   # shift_opsum, diag_mpo, exact_sum, MPSEvaluator
include(joinpath(@__DIR__, "fabric_common.jl"))

struct FGrid
    nbits::Vector{Int}                 # bits per dimension (space dims first, then mu, phi)
    order::Vector{Tuple{Int,Int}}      # chain order of (dim, k), k = 1 most significant bit
    sites::Vector{Index{Int}}
    pos::Vector{Vector{Int}}
end

function FGrid(nbits::Vector{Int}, order::Vector{Tuple{Int,Int}})
    sites = siteinds("S=1/2", length(order))
    pos = [zeros(Int, nbits[a]) for a in eachindex(nbits)]
    for (j, (a, k)) in enumerate(order)
        pos[a][k] = j
    end
    return FGrid(nbits, order, sites, pos)
end

function cores_to_mps(cores, sites)
    N = length(cores)
    links = [Index(size(cores[k], 3), "Link,l=$k") for k in 1:N-1]
    T = ITensor[]
    for k in 1:N
        c = cores[k]
        if N == 1
            push!(T, ITensor(reshape(c, size(c, 2)), sites[1]))
        elseif k == 1
            push!(T, ITensor(reshape(c, size(c, 2), size(c, 3)), sites[k], links[k]))
        elseif k == N
            push!(T, ITensor(reshape(c, size(c, 1), size(c, 2)), links[k-1], sites[k]))
        else
            push!(T, ITensor(c, links[k-1], sites[k], links[k]))
        end
    end
    return MPS(T)
end

# MPS of an array F over the dimensions ds (F has size (2^nbits[ds[1]], 2^nbits[ds[2]], ...)),
# constant along all other dimensions (identity cores carry the bond through their sites)
function embed_mps(F::AbstractArray, ds::Vector{Int}, g::FGrid; eps = 1e-15)
    nb = [g.nbits[a] for a in ds]
    ord = [(findfirst(==(a), ds), k) for (a, k) in g.order if a in ds]
    _, _, cores = tt_svd(quantics_tensor(F, nb, ord), eps; keep_cores = true)
    full = Array{Float64,3}[]
    r = 1; ci = 1
    for (a, k) in g.order
        if a in ds
            c = cores[ci]; ci += 1
            push!(full, c); r = size(c, 3)
        else
            c = zeros(r, 2, r)
            for i in 1:r, s in 1:2
                c[i, s, i] = 1.0
            end
            push!(full, c)
        end
    end
    return cores_to_mps(full, g.sites)
end

dense_to_mps(A::AbstractArray, g::FGrid; eps = 1e-12) = embed_mps(A, collect(1:length(g.nbits)), g; eps = eps)

function mps_to_dense(psi::MPS, g::FGrid)
    T = Array(contract(psi), g.sites...)
    p = quantics_perm(g.nbits, g.order)
    B = permutedims(T, invperm(p))
    return reshape(B, ntuple(a -> 2^g.nbits[a], length(g.nbits))...)
end

# point evaluation: idx = 0-based grid index per dimension
function fbits(g::FGrid, idx)
    sigma = zeros(Int, length(g.order))
    for (j, (a, k)) in enumerate(g.order)
        sigma[j] = ((idx[a] >> (g.nbits[a] - k)) & 1) + 1
    end
    return sigma
end

fshift(g::FGrid, a, dir; pbc = false) = MPO(shift_opsum(g.pos[a], dir; pbc = pbc), g.sites)
fdiag(g::FGrid, vals::AbstractArray, ds::Vector{Int}) = diag_mpo(embed_mps(vals, ds, g))
mprod(A::MPO, B::MPO; cutoff = 1e-28) = apply(A, B; alg = "naive", truncate = true, cutoff = cutoff)
mprod(A::MPO, B::MPO, C::MPO...; cutoff = 1e-28) = mprod(mprod(A, B; cutoff = cutoff), C...; cutoff = cutoff)
msum(coeffs, ops; cutoff = 1e-28) = truncate!(exact_sum([c * o for (c, o) in zip(coeffs, ops)]...); cutoff = cutoff)

# Orientation operators (identity on the space sites). amu, aphi = dimension numbers of mu and phi.
function sphere_ops(g::FGrid, amu, aphi; cutoff = 1e-28)
    Nmu, Nphi = 2^g.nbits[amu], 2^g.nbits[aphi]
    hmu, hphi = 2 / Nmu, 2pi / Nphi
    muf = [-1 + i * hmu for i in 1:Nmu]                 # face to the right of cell i
    muc = mu_grid(Nmu); phic = phi_grid(Nphi)
    phif = [j * hphi for j in 1:Nphi]
    Id = MPO(g.sites, "Id")
    Smp, Smm = fshift(g, amu, +1), fshift(g, amu, -1)
    Spp, Spm = fshift(g, aphi, +1; pbc = true), fshift(g, aphi, -1; pbc = true)
    Avgmu = msum([0.5, 0.5], [Id, Smp]); Divmu = msum([1 / hmu, -1 / hmu], [Id, Smm])
    Avgph = msum([0.5, 0.5], [Id, Spp]); Divph = msum([1 / hphi, -1 / hphi], [Id, Spm])
    Dmu_g = mprod(Divmu, fdiag(g, muf .* (1 .- muf .^ 2), [amu]), Avgmu; cutoff = cutoff)
    Dph(s) = mprod(Divph, fdiag(g, s.(phif), [aphi]), Avgph; cutoff = cutoff)
    Cph(c) = fdiag(g, c.(phic), [aphi])
    Oe = msum([1.0, 1.0], [mprod(Dmu_g, Cph(p -> cos(2p))), Dph(p -> sin(2p))]; cutoff = cutoff)
    Od = msum([1.0, -1.0], [mprod(Dmu_g, Cph(p -> sin(2p))), Dph(p -> cos(2p))]; cutoff = cutoff)
    Ow = msum([-1.0], [Dph(p -> 1.0)]; cutoff = cutoff)
    Lmu = mprod(Divmu, fdiag(g, 1 .- muf .^ 2, [amu]), msum([1 / hmu, -1 / hmu], [Smp, Id]); cutoff = cutoff)
    Lph = mprod(fdiag(g, 1 ./ (1 .- muc .^ 2), [amu]), msum([1 / hphi^2, -2 / hphi^2, 1 / hphi^2], [Spp, Id, Spm]); cutoff = cutoff)
    Lap = msum([1.0, 1.0], [Lmu, Lph]; cutoff = cutoff)
    return (Oe = Oe, Od = Od, Ow = Ow, Lap = Lap)
end

# Full operator: sum_k diag(a_k) O_k + lambda Lap; coefs = (e, d, w) arrays over the space dims sdims
function fabric_mpo(g::FGrid, coefs, sdims::Vector{Int}, amu, aphi, lambda; cutoff = 1e-26)
    O = sphere_ops(g, amu, aphi)
    terms = MPO[]
    for (a, Ok) in zip(coefs, (O.Oe, O.Od, O.Ow))
        maximum(abs, a) == 0 && continue
        push!(terms, mprod(fdiag(g, a, sdims), Ok; cutoff = cutoff))
    end
    lambda > 0 && push!(terms, lambda * O.Lap)
    return msum(ones(length(terms)), terms; cutoff = cutoff), O
end

# Classical RK4 in Horner form: v <- psi + (c dt) A v for c = 1/4, 1/3, 1/2, 1
function rk4_mps(psi::MPS, A::MPO, dt; cutoff = 1e-12, maxdim = 512, alg = "densitymatrix")
    v = psi
    for c in (1 / 4, 1 / 3, 1 / 2, 1.0)
        Av = apply(A, v; alg = alg, cutoff = cutoff, maxdim = maxdim)
        v = truncate!(exact_sum(psi, (c * dt) * Av); cutoff = cutoff, maxdim = maxdim)
    end
    return v
end

# ------------------------------------------------------------------------------------------------
# Dense reference with the same discretisation. psi is (Ns, Nmu, Nphi) with the space points flattened.
# ------------------------------------------------------------------------------------------------
struct DenseFabric
    e::Vector{Float64}; d::Vector{Float64}; w::Vector{Float64}
    lambda::Float64
    gf::Vector{Float64}; omf::Vector{Float64}; invc::Vector{Float64}    # mu face / centre coefficients
    c2c::Vector{Float64}; s2c::Vector{Float64}; c2f::Vector{Float64}; s2f::Vector{Float64}
    hmu::Float64; hphi::Float64
end

function DenseFabric(e, d, w, lambda, Nmu, Nphi)
    hmu, hphi = 2 / Nmu, 2pi / Nphi
    muf = [-1 + i * hmu for i in 1:Nmu]; muc = mu_grid(Nmu)
    phic = phi_grid(Nphi); phif = [j * hphi for j in 1:Nphi]
    return DenseFabric(vec(e), vec(d), vec(w), lambda, muf .* (1 .- muf .^ 2), 1 .- muf .^ 2, 1 ./ (1 .- muc .^ 2),
                       cos.(2phic), sin.(2phic), cos.(2phif), sin.(2phif), hmu, hphi)
end

function dense_rhs!(out, psi, P::DenseFabric)
    Ns, Nmu, Nphi = size(psi)
    Threads.@threads for j in 1:Nphi
        jp = j == Nphi ? 1 : j + 1; jm = j == 1 ? Nphi : j - 1
        @inbounds for i in 1:Nmu, r in 1:Ns
            e, d, w = P.e[r], P.d[r], P.w[r]
            p = psi[r, i, j]
            # mu fluxes  F_i = g_f (psi_i + psi_{i+1}) / 2 q(phi_j)
            q = e * P.c2c[j] + d * P.s2c[j]
            pip = i < Nmu ? psi[r, i+1, j] : 0.0
            pim = i > 1 ? psi[r, i-1, j] : 0.0
            Fr = P.gf[i] * (p + pip) / 2 * q
            Fl = i > 1 ? P.gf[i-1] * (pim + p) / 2 * q : 0.0
            # phi fluxes  V_f (psi_j + psi_{j+1}) / 2, V = w + d cos 2phi - e sin 2phi at the faces
            Vr = w + d * P.c2f[j] - e * P.s2f[j]
            Vl = w + d * P.c2f[jm] - e * P.s2f[jm]
            pjp = psi[r, i, jp]; pjm = psi[r, i, jm]
            rhs = (Fr - Fl) / P.hmu - (Vr * (p + pjp) / 2 - Vl * (pjm + p) / 2) / P.hphi
            if P.lambda > 0
                Dr = P.omf[i] * (pip - p)
                Dl = i > 1 ? P.omf[i-1] * (p - pim) : 0.0
                rhs += P.lambda * ((Dr - Dl) / P.hmu^2 + P.invc[i] * (pjp - 2p + pjm) / P.hphi^2)
            end
            out[r, i, j] = rhs
        end
    end
    return out
end

function rk4_dense!(psi, P::DenseFabric, dt, buf)
    k, v = buf
    v .= psi
    for c in (1 / 4, 1 / 3, 1 / 2, 1.0)
        dense_rhs!(k, v, P)
        @. v = psi + (c * dt) * k
    end
    psi .= v
    return psi
end
