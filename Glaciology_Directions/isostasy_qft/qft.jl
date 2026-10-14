# qft.jl -- quantics grids, the quantum Fourier transform (QFT) as an MPO, Fourier multipliers as
# MPS, and the helpers used by the isostasy study (Glaciology_Directions/isostasy_qft).
#
# Conventions
#   A field on a (2^n)^d grid is an MPS over d*n binary sites; site state 1 <-> bit 0, 2 <-> bit 1.
#   g.pos[a][k] is the MPS site that holds the k-th bit of dimension a.
#   Real space ("normal"):   the bit at pos[a][k] has weight 2^(n-k)   (k = 1 is the MSB).
#   Fourier space ("rev"):   the bit at pos[a][k] has weight 2^(k-1)   (k = 1 is the LSB).
#   The QFT circuit transforms each bit in place, so its output is bit reversed (Chen, Stoudenmire &
#   White, PRX Quantum 4, 040318 (2023)): with this ordering the QFT MPO has a small bond dimension.
#   Fourier coefficients follow FFTW: X[m] = sum_j x[j] exp(-2 pi i j m / N), and the inverse is
#   Q^dagger / N^d. The QFT output (rev) is the input of the inverse, so no bit swaps are ever needed.
#
#   Dense arrays A[i1+1, ..., id+1] are converted with array_to_mps / mps_to_array (exact SVD, only
#   for moderate sizes); every conversion is checked by point evaluation in the tests.

using ITensors, ITensorMPS, LinearAlgebra, Printf, Random
import TensorCrossInterpolation as TCI
using TCIITensorConversion
ITensors.disable_warn_order()

struct QGrid
    d::Int
    n::Int
    ordering::Symbol
    sites::Vector{Index{Int}}
    pos::Vector{Vector{Int}}
end

function QGrid(d::Int, n::Int; ordering::Symbol = :interleaved, sites = nothing)
    pos = ordering == :interleaved ? [[(k - 1) * d + a for k in 1:n] for a in 1:d] :
          ordering == :sequential  ? [[(a - 1) * n + k for k in 1:n] for a in 1:d] :
          error("ordering must be :interleaved or :sequential")
    sites === nothing && (sites = [Index(2, "Qubit,Site,n=$j") for j in 1:d*n])
    return QGrid(d, n, ordering, sites, pos)
end

nsites(g::QGrid) = g.d * g.n
npts(g::QGrid) = 2^g.n
bitexp(g::QGrid, k, rev) = rev ? k - 1 : g.n - k

function bits_to_idx(g::QGrid, sigma; rev = false)
    return ntuple(g.d) do a
        i = 0
        for k in 1:g.n
            i += (sigma[g.pos[a][k]] - 1) << bitexp(g, k, rev)
        end
        i
    end
end

function idx_to_bits(g::QGrid, idx; rev = false)
    sigma = zeros(Int, nsites(g))
    for a in 1:g.d, k in 1:g.n
        sigma[g.pos[a][k]] = ((idx[a] >> bitexp(g, k, rev)) & 1) + 1
    end
    return sigma
end

# reshape(A, 2, ..., 2) (column major): the bit with exponent e of dimension a is array dim (a-1)n+e+1
function site_perm(g::QGrid; rev = false)
    perm = zeros(Int, nsites(g))
    for a in 1:g.d, k in 1:g.n
        perm[g.pos[a][k]] = (a - 1) * g.n + bitexp(g, k, rev) + 1
    end
    return perm
end

function array_to_mps(A::AbstractArray, g::QGrid; rev = false, cutoff = 1e-15, maxdim = typemax(Int))
    T = permutedims(reshape(collect(A), ntuple(_ -> 2, nsites(g))), site_perm(g; rev = rev))
    return MPS(T, g.sites; cutoff = cutoff, maxdim = maxdim)
end

function mps_to_array(psi::MPS, g::QGrid; rev = false)
    T = Array(prod(psi), g.sites...)
    A = permutedims(T, invperm(site_perm(g; rev = rev)))
    return reshape(A, ntuple(_ -> npts(g), g.d))
end

# Fast point evaluation (plain arrays)
struct MPSEval{T}
    cores::Vector{Array{T,3}}
end

function MPSEval(psi::MPS)
    N = length(psi)
    T = ComplexF64
    l = linkinds(psi)
    cores = Array{T,3}[]
    for k in 1:N
        s = siteind(psi, k)
        A = psi[k]
        if N == 1
            push!(cores, reshape(Array{T}(A, s), 1, 2, 1))
        elseif k == 1
            push!(cores, reshape(Array{T}(A, s, l[1]), 1, 2, dim(l[1])))
        elseif k == N
            push!(cores, reshape(Array{T}(A, l[N-1], s), dim(l[N-1]), 2, 1))
        else
            push!(cores, Array{T}(A, l[k-1], s, l[k]))
        end
    end
    return MPSEval{T}(cores)
end

function (E::MPSEval)(sigma)
    v = transpose(E.cores[1][1, sigma[1], :])
    for k in 2:length(E.cores)
        v = v * @view E.cores[k][:, sigma[k], :]
    end
    return v[1]
end

evaluate(E::MPSEval, g::QGrid, idx; rev = false) = E(idx_to_bits(g, idx; rev = rev))

# ---------------------------------------------------------------------------------------------
# MPS algebra
# ---------------------------------------------------------------------------------------------

nparams(psi::Union{MPS,MPO}) = sum(dim, psi)
exact_sum(xs...) = length(xs) == 1 ? copy(xs[1]) : +(xs...; alg = "directsum")
lincomb(cs, xs; kwargs...) = truncate!(exact_sum([c * x for (c, x) in zip(cs, xs)]...); kwargs...)

function diag_mpo(m::MPS)
    W = MPO(length(m))
    for k in 1:length(m)
        s = siteind(m, k)
        W[k] = m[k] * delta(s, s', s'')
    end
    setprime!(W, 0; :plev => 2)
    return W
end

# elementwise product m .* psi
hadamard(m::MPS, psi::MPS; alg = "naive", kwargs...) = apply(diag_mpo(m), psi; alg = alg, kwargs...)

# product state with every entry equal to c
function constant_mps(g::QGrid, c)
    N = nsites(g)
    links = [Index(1, "Link,l=$k") for k in 1:N-1]
    ts = ITensor[]
    for k in 1:N
        s = g.sites[k]
        is = k == 1 ? (s, links[1]) : k == N ? (s, links[N-1]) : (s, links[k-1], links[k])
        T = ITensor(ComplexF64, is...)
        for st in 1:2
            T[s => st, [l => 1 for l in is[2:end]]...] = k == 1 ? ComplexF64(c) : 1.0
        end
        push!(ts, T)
    end
    return MPS(ts)
end

# ---------------------------------------------------------------------------------------------
# Tensor cross interpolation of a function of the grid index (0-based tuple)
# ---------------------------------------------------------------------------------------------

function _pivot_candidates(N)
    c = unique(vcat(0:min(N - 1, 16), max(0, N - 16):N-1, round.(Int, range(0, N - 1, length = min(N, 64)))))
    return c
end

function tci_mps(f, g::QGrid; rev = false, T = Float64, tolerance = 1e-12, maxbonddim = 400,
                 pivots = Vector{Int}[], verbosity = 0, recompress = 0.0)
    F(sigma) = T(f(bits_to_idx(g, sigma; rev = rev)))
    # starting pivots: largest |f| over a coarse candidate set, globally and in every orthant
    # (lower/upper half of each index range). Fourier multipliers live near k = 0, i.e. in the four
    # corners of the index square; a single pivot lets TCI miss the other corners at large n.
    cand = _pivot_candidates(npts(g))
    N = npts(g)
    best = Dict{NTuple{g.d,Bool},Tuple{Any,Float64}}()
    for idx in Iterators.product(ntuple(_ -> cand, g.d)...)
        v = abs(f(idx))
        key = ntuple(a -> idx[a] >= N ÷ 2, g.d)
        (!haskey(best, key) || v > best[key][2]) && (best[key] = (idx, v))
    end
    vmax = maximum(last(b) for b in values(best))
    piv = [idx_to_bits(g, b[1]; rev = rev) for b in sort(collect(values(best)), by = x -> -x[2]) if b[2] > 1e-3 * vmax]
    tci, ranks, errors = TCI.crossinterpolate2(T, F, fill(2, nsites(g)), [piv; pivots];
                                               tolerance = tolerance, maxbonddim = maxbonddim,
                                               verbosity = verbosity)
    psi = MPS(tci; sites = g.sites)
    recompress > 0 && truncate!(psi; cutoff = recompress)
    return psi, maximum(ranks), errors[end]
end

# ---------------------------------------------------------------------------------------------
# QFT MPO (circuit construction with SVD compression after every layer)
# ---------------------------------------------------------------------------------------------

const HMAT = ComplexF64[1 1; 1 -1]   # unnormalised Hadamard: X[m] = sum_j x_j w^{jm} exactly

function _apply_h!(Q::MPO, s::Index)
    j = findfirst(t -> hasind(t, s), Q)
    Qj = replaceind(Q[j], s', s'')
    Q[j] = ITensor(HMAT, s', s'') * Qj
    return Q
end

# Diagonal MPO: prod_{l>k} exp(i sign pi b_c b_l / 2^(l-k)), c = pos[a][k], bond 2 from c to pos[a][n]
function phase_layer(g::QGrid, a::Int, k::Int, sign::Int)
    N = nsites(g)
    c, last = g.pos[a][k], g.pos[a][g.n]
    theta = Dict(g.pos[a][l] => sign * pi / 2^(l - k) for l in k+1:g.n)
    links = [Index((c <= j < last) ? 2 : 1, "Link,l=$j") for j in 1:N-1]
    W = MPO(N)
    for j in 1:N
        s = g.sites[j]
        lL = j > 1 ? links[j-1] : nothing
        lR = j < N ? links[j] : nothing
        is = Index[s', dag(s)]
        lL === nothing || push!(is, lL)
        lR === nothing || push!(is, lR)
        T = ITensor(ComplexF64, is...)
        cL = lL === nothing ? 1 : dim(lL)
        cR = lR === nothing ? 1 : dim(lR)
        for x in 0:1, bl in 1:cL, br in 1:cR
            if j < c || j > last
                v = 1.0 + 0im
            elseif j == c
                v = br == x + 1 ? 1.0 + 0im : 0.0im
            elseif j < last
                v = bl == br ? (haskey(theta, j) ? cis(theta[j] * (bl - 1) * x) : 1.0 + 0im) : 0.0im
            else
                v = cis(theta[j] * (bl - 1) * x)
            end
            v == 0 && continue
            ass = Pair{Index{Int},Int}[s' => x + 1, s => x + 1]
            lL === nothing || push!(ass, lL => bl)
            lR === nothing || push!(ass, lR => br)
            T[ass...] = v
        end
        W[j] = T
    end
    return W
end

"""
    qft_mpo(g; sign = -1, cutoff = 1e-14, maxdim)

MPO of the d-dimensional DFT on grid g: input bits in real-space order (MSB first), output bits
reversed in place (LSB first). sign = -1 is FFTW's forward transform (unnormalised).
"""
function qft_mpo(g::QGrid; sign::Int = -1, cutoff = 1e-14, maxdim = typemax(Int), dims = 1:g.d)
    Q = MPO(ComplexF64, g.sites, "Id")
    for a in dims, k in 1:g.n
        _apply_h!(Q, g.sites[g.pos[a][k]])
        if k < g.n
            Q = apply(phase_layer(g, a, k, sign), Q; alg = "naive", cutoff = cutoff, maxdim = maxdim)
        end
    end
    truncate!(Q; cutoff = cutoff, maxdim = maxdim)
    return Q
end

# Inverse transform (unnormalised): Q^dagger maps rev-ordered Fourier coefficients to real space.
# The caller divides by N^d (folded into the multiplier).
adjoint_mpo(Q::MPO) = swapprime(dag(Q), 0 => 1)

# QFT applied gate by gate to an MPS (no MPO), truncating after every phase layer
function qft_circuit(psi::MPS, g::QGrid; sign::Int = -1, cutoff = 1e-14, maxdim = typemax(Int))
    phi = copy(psi)
    for a in 1:g.d, k in 1:g.n
        j = g.pos[a][k]
        s = g.sites[j]
        phi[j] = noprime(ITensor(HMAT, s', s) * phi[j])
        if k < g.n
            phi = apply(phase_layer(g, a, k, sign), phi; alg = "naive", cutoff = cutoff, maxdim = maxdim)
        end
    end
    return phi
end

# ---------------------------------------------------------------------------------------------
# Wavenumbers
# ---------------------------------------------------------------------------------------------

msigned(m, N) = m < N ÷ 2 ? m : m - N
fftk(N, L) = (2pi / L) .* [msigned(m, N) for m in 0:N-1]

# |k| for a d-tuple of 0-based Fourier indices
kabs(idx, N, L) = (2pi / L) * sqrt(sum(abs2(msigned(m, N)) for m in idx))

timed(f) = (t = time(); r = f(); (r, time() - t))

# Product-state MPS with site vectors vecs[j] (length 2) on site j
function product_mps(g::QGrid, vecs)
    N = nsites(g)
    links = [Index(1, "Link,l=$k") for k in 1:N-1]
    ts = ITensor[]
    for k in 1:N
        s = g.sites[k]
        is = N == 1 ? (s,) : k == 1 ? (s, links[1]) : k == N ? (s, links[N-1]) : (s, links[k-1], links[k])
        T = ITensor(ComplexF64, is...)
        for st in 1:2
            T[s => st, [l => 1 for l in is[2:end]]...] = vecs[k][st]
        end
        push!(ts, T)
    end
    return MPS(ts)
end

# conj of the inverse-DFT phase e^{2 pi i m.j / N} as a product state over the rev (Fourier) bits,
# so that inner(phase_state(g, j), uh) = sum_m e^{2 pi i m.j/N} uh[m] = N^d * u[j]
function phase_state(g::QGrid, j)
    N = npts(g)
    vecs = Vector{Vector{ComplexF64}}(undef, nsites(g))
    for a in 1:g.d, k in 1:g.n
        vecs[g.pos[a][k]] = [1.0, cis(-2pi * mod(2^(k - 1) * j[a], N) / N)]
    end
    return product_mps(g, vecs)
end

# |m = 0> in the Fourier (rev) order: every bit 0
zero_mode_state(g::QGrid) = product_mps(g, [ComplexF64[1, 0] for _ in 1:nsites(g)])
