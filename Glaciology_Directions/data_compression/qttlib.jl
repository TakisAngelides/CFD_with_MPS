# qttlib.jl -- quantics tensor-train (QTT) compression of 2D rasters with plain Julia arrays,
# plus the baselines it is compared with (matrix SVD, Haar / CDF 9/7 wavelets, quantised zlib)
# and the preprocessing helpers (harmonic fill of masked cells, block-mean coarsening).
#
# Conventions (identical to Glaciology_SIA/quantics.jl):
#   field A[i+1, j+1], i = x index, j = y index, 0 <= i, j < N = 2^n
#   bit k = 1 is the most significant bit of an index
#   :interleaved  chain = x1 y1 x2 y2 ... xn yn
#   :sequential   chain = x1 ... xn y1 ... yn
# Here the chain tensor is stored as a plain vector with site 1 the FASTEST index (column-major),
# cores are (r_{k-1}, 2, r_k) arrays, site state 1 <-> bit 0, state 2 <-> bit 1.

using LinearAlgebra, Statistics, Printf
using Zlib_jll

# ---------------------------------------------------------------------------------------------
# Bit layouts
# ---------------------------------------------------------------------------------------------

# chain positions (1-based) of x bit k and y bit k, k = 1 most significant
function site_positions(n::Int, ordering::Symbol)
    if ordering == :interleaved
        return [2(k - 1) + 1 for k in 1:n], [2k for k in 1:n]
    elseif ordering == :sequential
        return collect(1:n), collect(n+1:2n)
    end
    error("unknown ordering $ordering")
end

# 0-based chain linear index = Lx[i+1] + Ly[j+1]
function layout_tables(n::Int, ordering::Symbol)
    px, py = site_positions(n, ordering)
    Lx = zeros(Int, 2^n); Ly = zeros(Int, 2^n)
    for i in 0:2^n-1, k in 1:n
        b = (i >> (n - k)) & 1
        Lx[i+1] += b << (px[k] - 1)
        Ly[i+1] += b << (py[k] - 1)
    end
    return Lx, Ly
end

nbits(A::AbstractMatrix) = (N = size(A, 1); n = round(Int, log2(N)); (size(A) == (N, N) && 2^n == N) || error("need a 2^n x 2^n array"); n)

function to_chain(A::AbstractMatrix, ordering::Symbol)
    n = nbits(A); N = 2^n
    Lx, Ly = layout_tables(n, ordering)
    T = Vector{Float64}(undef, N * N)
    @inbounds for j in 1:N
        ly = Ly[j]
        for i in 1:N
            T[Lx[i] + ly + 1] = A[i, j]
        end
    end
    return T
end

function from_chain(T::AbstractVector, n::Int, ordering::Symbol)
    N = 2^n
    Lx, Ly = layout_tables(n, ordering)
    A = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N
        ly = Ly[j]
        for i in 1:N
            A[i, j] = T[Lx[i] + ly + 1]
        end
    end
    return A
end

# ---------------------------------------------------------------------------------------------
# Tensor train by sequential SVD (TT-SVD, Oseledets 2011)
# ---------------------------------------------------------------------------------------------

struct QTT
    cores::Vector{Array{Float64,3}}
end

qtt_params(q::QTT) = sum(length, q.cores)
qtt_ranks(q::QTT) = [size(c, 3) for c in q.cores[1:end-1]]

# smallest r with sum_{i > r} s_i^2 <= delta2
function truncation_rank(s::AbstractVector, delta2::Real)
    tail = 0.0; r = length(s)
    while r > 1 && tail + s[r]^2 <= delta2
        tail += s[r]^2
        r -= 1
    end
    return r
end

function safe_svd(M)
    try
        return svd(M)
    catch
        return svd(M; alg = LinearAlgebra.QRIteration())
    end
end

# TT-SVD of a chain vector T (length 2^L) with global relative Frobenius error <= eps:
# every bond discards at most eps^2 ||T||^2 / (L-1). Returns `nothing` if the parameter count
# exceeds `maxparams` (no point continuing: worse than storing the data).
function ttsvd(T::AbstractVector{Float64}; eps::Real = 1e-3, maxparams::Real = Inf, maxrank::Int = typemax(Int))
    L = round(Int, log2(length(T)))
    delta2 = eps^2 * sum(abs2, T) / (L - 1)
    cores = Array{Float64,3}[]
    C = T
    r = 1
    rest = length(T)
    np = 0
    for k in 1:L-1
        rest ÷= 2
        M = reshape(C, r * 2, rest)
        F = safe_svd(M)
        rn = min(truncation_rank(F.S, delta2), maxrank)
        push!(cores, reshape(F.U[:, 1:rn], r, 2, rn))
        np += 2r * rn
        np > maxparams && return nothing
        C = vec(Diagonal(F.S[1:rn]) * F.Vt[1:rn, :])
        r = rn
    end
    push!(cores, reshape(C, r, 2, 1))
    return QTT(cores)
end

# cores[1:m] -> matrix 2^m x r_m (rows = chain index of sites 1..m, site 1 fastest)
function contract_left(cores)
    W = reshape(cores[1], 2, :)
    for k in 2:length(cores)
        c = cores[k]; r0, _, r1 = size(c)
        W = reshape(W * reshape(c, r0, 2r1), :, r1)
    end
    return W
end

# cores -> matrix r_0 x 2^m (columns = chain index of these sites, first one fastest)
function contract_right(cores)
    c = cores[end]
    W = reshape(c, size(c, 1), 2)
    for k in length(cores)-1:-1:1
        c = cores[k]; r0, _, r1 = size(c)
        W = reshape(reshape(c, r0 * 2, r1) * W, r0, :)
    end
    return W
end

function qtt_full(q::QTT)
    L = length(q.cores); m = L ÷ 2
    return vec(contract_left(q.cores[1:m]) * contract_right(q.cores[m+1:end]))
end

qtt_to_array(q::QTT, n::Int, ordering::Symbol) = from_chain(qtt_full(q), n, ordering)

# value at 0-based grid point (i, j) directly from the cores (independent check of the layout)
function qtt_eval(q::QTT, i::Int, j::Int, n::Int, ordering::Symbol)
    px, py = site_positions(n, ordering)
    s = zeros(Int, 2n)
    for k in 1:n
        s[px[k]] = (i >> (n - k)) & 1
        s[py[k]] = (j >> (n - k)) & 1
    end
    v = ones(1, 1)
    for p in 1:2n
        v = v * q.cores[p][:, s[p]+1, :]
    end
    return v[1]
end

# ---------------------------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------------------------

function errstats(A::AbstractMatrix, B::AbstractMatrix, mask::AbstractMatrix{Bool})
    s2all = 0.0; s2m = 0.0; mx = 0.0; nm = 0
    @inbounds for k in eachindex(A)
        e = A[k] - B[k]
        s2all += e^2
        if mask[k]
            s2m += e^2; nm += 1
            mx = max(mx, abs(e))
        end
    end
    return (rms_all = sqrt(s2all / length(A)), rms = sqrt(s2m / max(nm, 1)), maxerr = mx)
end

# ---------------------------------------------------------------------------------------------
# Wavelets: orthonormal Haar and CDF 9/7 (lifting, symmetric extension, near-orthonormal scaling)
# ---------------------------------------------------------------------------------------------

const α97 = -1.586134342059924
const β97 = -0.052980118572961
const γ97 = 0.882911075530934
const δ97 = 0.443506852043971
const ζ97 = 1.149604398860241

function cdf97_fwd!(x::AbstractVector, buf::Vector{Float64})
    m = length(x); h = m ÷ 2
    @inbounds begin
        for i in 1:h; buf[i] = x[2i-1]; buf[h+i] = x[2i]; end
        for i in 1:h; buf[h+i] += α97 * (buf[i] + buf[min(i + 1, h)]); end
        for i in 1:h; buf[i] += β97 * (buf[h+max(i - 1, 1)] + buf[h+i]); end
        for i in 1:h; buf[h+i] += γ97 * (buf[i] + buf[min(i + 1, h)]); end
        for i in 1:h; buf[i] += δ97 * (buf[h+max(i - 1, 1)] + buf[h+i]); end
        for i in 1:h; x[i] = buf[i] * ζ97; x[h+i] = buf[h+i] / ζ97; end
    end
    return x
end

function cdf97_inv!(x::AbstractVector, buf::Vector{Float64})
    m = length(x); h = m ÷ 2
    @inbounds begin
        for i in 1:h; buf[i] = x[i] / ζ97; buf[h+i] = x[h+i] * ζ97; end
        for i in 1:h; buf[i] -= δ97 * (buf[h+max(i - 1, 1)] + buf[h+i]); end
        for i in 1:h; buf[h+i] -= γ97 * (buf[i] + buf[min(i + 1, h)]); end
        for i in 1:h; buf[i] -= β97 * (buf[h+max(i - 1, 1)] + buf[h+i]); end
        for i in 1:h; buf[h+i] -= α97 * (buf[i] + buf[min(i + 1, h)]); end
        for i in 1:h; x[2i-1] = buf[i]; x[2i] = buf[h+i]; end
    end
    return x
end

function haar_fwd!(x::AbstractVector, buf::Vector{Float64})
    m = length(x); h = m ÷ 2; c = 1 / sqrt(2)
    @inbounds for i in 1:h
        buf[i] = c * (x[2i-1] + x[2i]); buf[h+i] = c * (x[2i-1] - x[2i])
    end
    @inbounds for i in 1:m; x[i] = buf[i]; end
    return x
end

function haar_inv!(x::AbstractVector, buf::Vector{Float64})
    m = length(x); h = m ÷ 2; c = 1 / sqrt(2)
    @inbounds for i in 1:h
        buf[2i-1] = c * (x[i] + x[h+i]); buf[2i] = c * (x[i] - x[h+i])
    end
    @inbounds for i in 1:m; x[i] = buf[i]; end
    return x
end

# Mallat 2D transform in place, `levels` levels (columns then rows at each level)
function wt2d!(A::Matrix{Float64}, levels::Int, f1!)
    N = size(A, 1)
    buf = zeros(N); row = zeros(N)
    m = N
    for _ in 1:levels
        Threads.@threads for j in 1:m
            f1!(view(A, 1:m, j), zeros(m))
        end
        Threads.@threads for i in 1:m
            r = A[i, 1:m]
            f1!(r, zeros(m))
            A[i, 1:m] = r
        end
        m ÷= 2
    end
    return A
end

function iwt2d!(A::Matrix{Float64}, levels::Int, f1inv!)
    N = size(A, 1)
    for l in levels:-1:1
        m = N >> (l - 1)
        Threads.@threads for i in 1:m
            r = A[i, 1:m]
            f1inv!(r, zeros(m))
            A[i, 1:m] = r
        end
        Threads.@threads for j in 1:m
            f1inv!(view(A, 1:m, j), zeros(m))
        end
    end
    return A
end

# keep the K largest-magnitude coefficients (threshold = K-th largest |c|)
function keep_largest(C::Matrix{Float64}, K::Int, sorted_abs::Vector{Float64})
    K >= length(C) && return copy(C)
    thr = sorted_abs[K]
    return map(c -> abs(c) >= thr ? c : 0.0, C)
end

# lower bound on the bytes needed to store the positions of K out of N coefficients
sigmap_bytes(K, N) = (p = K / N; p <= 0 || p >= 1 ? 0.0 : N * (-p * log2(p) - (1 - p) * log2(1 - p)) / 8)

# ---------------------------------------------------------------------------------------------
# zlib of uniformly quantised data (lossy with max error step/2), previous-row prediction
# ---------------------------------------------------------------------------------------------

function zlib_bytes(data::Vector{UInt8}; level::Int = 6)
    bound = ccall((:compressBound, libz), Culong, (Culong,), length(data))
    out = Vector{UInt8}(undef, bound)
    outlen = Ref{Culong}(bound)
    rc = ccall((:compress2, libz), Cint, (Ptr{UInt8}, Ref{Culong}, Ptr{UInt8}, Culong, Cint),
               out, outlen, data, length(data), level)
    rc == 0 || error("zlib compress2 error $rc")
    return Int(outlen[])
end

# quantise with step Δ, predict each value from its x-neighbour, store in the narrowest int type
function quant_zlib(A::Matrix{Float64}, Δ::Real; level::Int = 6)
    q = round.(Int64, A ./ Δ)
    d = similar(q)
    @inbounds for j in axes(q, 2)
        d[1, j] = q[1, j]
        for i in 2:size(q, 1)
            d[i, j] = q[i, j] - q[i-1, j]
        end
    end
    lo, hi = extrema(d)
    T = (lo >= typemin(Int8) && hi <= typemax(Int8)) ? Int8 :
        (lo >= typemin(Int16) && hi <= typemax(Int16)) ? Int16 : Int32
    bytes = zlib_bytes(collect(reinterpret(UInt8, vec(T.(d)))); level = level)
    return bytes, q .* Δ, T
end

# ---------------------------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------------------------

coarsen2(A::AbstractMatrix) = 0.25 .* (A[1:2:end, 1:2:end] .+ A[2:2:end, 1:2:end] .+ A[1:2:end, 2:2:end] .+ A[2:2:end, 2:2:end])

# cell-centred bilinear upsampling m x m -> 2m x 2m (constant extension at the edges)
function upsample_bilinear(F::Matrix{Float64})
    m = size(F, 1); M = 2m
    G = Matrix{Float64}(undef, M, M)
    @inbounds for J in 1:M, I in 1:M
        x = (I - 0.5) / 2 + 0.5; y = (J - 0.5) / 2 + 0.5     # position in coarse cell-index units
        i0 = clamp(floor(Int, x), 1, m); j0 = clamp(floor(Int, y), 1, m)
        i1 = min(i0 + 1, m); j1 = min(j0 + 1, m)
        wx = clamp(x - i0, 0.0, 1.0); wy = clamp(y - j0, 0.0, 1.0)
        G[I, J] = (1 - wx) * (1 - wy) * F[i0, j0] + wx * (1 - wy) * F[i1, j0] + (1 - wx) * wy * F[i0, j1] + wx * wy * F[i1, j1]
    end
    return G
end

# Smooth extension of A from the `valid` cells into the rest (push-pull pyramid with bilinear
# prolongation and Jacobi smoothing of the unknown cells on every level, i.e. a multigrid-style
# approximately harmonic inpainting). Valid cells are returned unchanged.
function harmonic_fill(A::AbstractMatrix, valid::AbstractMatrix{Bool}; sweeps::Int = 10)
    V = [Float64.(ifelse.(valid, A, 0.0))]
    W = [Float64.(valid)]
    while size(V[end], 1) > 1
        push!(V, 4 .* coarsen2(V[end])); push!(W, 4 .* coarsen2(W[end]))
    end
    F = V[end] ./ max.(W[end], 1e-300)
    for l in length(V)-1:-1:1
        Fu = upsample_bilinear(F)
        known = W[l] .> 0
        Fl = ifelse.(known, V[l] ./ max.(W[l], 1e-300), Fu)
        m = size(Fl, 1)
        unk = findall(.!known)
        if !isempty(unk) && m > 2
            for _ in 1:sweeps
                G = copy(Fl)
                @inbounds for c in unk
                    i, j = c[1], c[2]
                    G[i, j] = 0.25 * (Fl[max(i - 1, 1), j] + Fl[min(i + 1, m), j] + Fl[i, max(j - 1, 1)] + Fl[i, min(j + 1, m)])
                end
                Fl = G
            end
        end
        F = Fl
    end
    return F
end
