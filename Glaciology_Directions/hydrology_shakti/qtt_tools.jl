# Plain-array quantics tensor-train (QTT) tools for compressibility studies of 2D fields.
# No ITensors here: exact TT-SVD sweeps on dense 2^n x 2^n arrays, with control over the ordering of
# the 2n binary sites, plus quadtree (AMR proxy) counts for the same fields.
#
# Conventions: A[i, j], i = x index (1..2^n), j = y index; bit 1 = most significant.
#   :interleaved -> sites x1 y1 x2 y2 ... xn yn
#   :sequential  -> sites x1 ... xn y1 ... yn
# A column-major reshape(A, 2, ..., 2) has dims [x_lsb..x_msb, y_lsb..y_msb], so the site order is a
# permutation of those 2n dims.

using LinearAlgebra, Printf, Statistics

function site_perm(n::Int, ordering::Symbol)
    xd(k) = n - k + 1          # dim of the k-th most significant x bit
    yd(k) = 2n - k + 1
    if ordering == :interleaved
        return vcat([[xd(k), yd(k)] for k in 1:n]...)
    elseif ordering == :sequential
        return vcat([xd(k) for k in 1:n], [yd(k) for k in 1:n])
    elseif ordering == :sequential_yx
        return vcat([yd(k) for k in 1:n], [xd(k) for k in 1:n])
    else
        error("unknown ordering $ordering")
    end
end

to_tensor(A::AbstractMatrix, ordering::Symbol) = begin
    n = round(Int, log2(size(A, 1)))
    @assert size(A) == (2^n, 2^n) "need a 2^n x 2^n array"
    permutedims(reshape(Array{Float64}(A), fill(2, 2n)...), site_perm(n, ordering))
end

from_tensor(T::AbstractArray, n::Int, ordering::Symbol) =
    reshape(permutedims(T, invperm(site_perm(n, ordering))), 2^n, 2^n)

struct TT
    cores::Vector{Array{Float64,3}}   # (r_{k-1}, 2, r_k)
end
ranks(tt::TT) = [size(c, 3) for c in tt.cores[1:end-1]]
nparams(tt::TT) = sum(length, tt.cores)
maxrank(tt::TT) = maximum(ranks(tt); init = 1)

# Left-to-right TT-SVD. Per-bond truncation: discard the smallest singular values whose root-sum-square
# is <= delta (absolute). Returns the TT and the singular-value spectrum at every bond (exact
# unfolding spectra if delta is tiny).
function tt_svd(T::AbstractArray; delta::Float64 = 0.0, maxrank::Int = typemax(Int))
    L = ndims(T)
    cores = Array{Float64,3}[]
    spectra = Vector{Float64}[]
    C = reshape(Array{Float64}(T), 2, :)
    r = 1
    for k in 1:L-1
        C = reshape(C, 2r, :)
        F = svd(C)
        S = F.S
        push!(spectra, copy(S))
        tail = sqrt.(reverse(cumsum(reverse(S .^ 2))))   # tail[j] = rss of S[j:end]
        rk = findfirst(t -> t <= delta, tail)
        rk = rk === nothing ? length(S) : max(rk - 1, 1)
        rk = min(rk, maxrank)
        push!(cores, reshape(F.U[:, 1:rk], r, 2, rk))
        C = Diagonal(S[1:rk]) * F.Vt[1:rk, :]
        r = rk
    end
    push!(cores, reshape(C, r, 2, 1))
    return TT(cores), spectra
end

function tt_full(tt::TT)
    X = reshape(tt.cores[1], 2, :)                     # (2, r1)
    for k in 2:length(tt.cores)
        c = tt.cores[k]
        r, _, rn = size(c)
        X = reshape(X * reshape(c, r, 2rn), :, rn)      # first sites fastest (column major)
    end
    return reshape(X, fill(2, length(tt.cores))...)
end

# eps-rank of every bond from exact spectra: smallest r with rss(S[r+1:end]) <= eps * ||F||
function eps_ranks(spectra, nrm, eps)
    map(spectra) do S
        tail = sqrt.(reverse(cumsum(reverse(S .^ 2))))
        r = findfirst(t -> t <= eps * nrm, tail)
        r === nothing ? length(S) : max(r - 1, 1)
    end
end

# Truncated QTT with total relative Frobenius error <= eps; returns (tt, rel_fro_err, rel_max_err)
function qtt_compress(A::AbstractMatrix, ordering::Symbol, eps::Float64)
    n = round(Int, log2(size(A, 1)))
    T = to_tensor(A, ordering)
    nrm = norm(T)
    tt, _ = tt_svd(T; delta = eps * nrm / sqrt(2n - 1))
    Ar = from_tensor(tt_full(tt), n, ordering)
    rng = maximum(A) - minimum(A)
    return tt, norm(Ar - A) / nrm, maximum(abs, Ar - A) / max(rng, eps * 1e-300)
end

# ---------------------------------------------------------------------------------------------
# AMR proxies
# ---------------------------------------------------------------------------------------------

# Error-controlled quadtree (AMR proxy): every leaf is one AMR cell carrying a bilinear
# reconstruction from the field values at its four VERTICES (vertex value = mean of the <= 4 cells
# sharing that vertex, i.e. the field itself, not a fit). A dyadic block is a leaf when that
# reconstruction reproduces every fine cell in the block to within `tol` (absolute); otherwise it is
# split. Single fine cells are always leaves. #leaves = number of AMR cells (DOFs) needed.
# Returns (leaves, leaves of one fine cell, leaf-size histogram Dict(size => count), reconstruction).
function quadtree_bilinear(A::AbstractMatrix, tol)
    N = size(A, 1)
    V = zeros(N + 1, N + 1)
    for b in 1:N+1, a in 1:N+1
        is = unique(clamp.((a - 1, a), 1, N)); js = unique(clamp.((b - 1, b), 1, N))
        V[a, b] = sum(A[i, j] for i in is, j in js) / (length(is) * length(js))
    end
    Ar = similar(A, Float64)
    leaves = Ref(0)
    hist = Dict{Int,Int}()
    function rec(i0, j0, s)
        if s == 1
            Ar[i0, j0] = A[i0, j0]; leaves[] += 1; hist[1] = get(hist, 1, 0) + 1
            return
        end
        i1, j1 = i0 + s - 1, j0 + s - 1
        f00, f10, f01, f11 = V[i0, j0], V[i1+1, j0], V[i0, j1+1], V[i1+1, j1+1]
        ok = true
        @inbounds for j in j0:j1, i in i0:i1
            tx = (i - i0 + 0.5) / s; ty = (j - j0 + 0.5) / s
            v = (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11
            if abs(v - A[i, j]) > tol
                ok = false; break
            end
            Ar[i, j] = v
        end
        if ok
            leaves[] += 1; hist[s] = get(hist, s, 0) + 1
        else
            h = s ÷ 2
            rec(i0, j0, h); rec(i0 + h, j0, h); rec(i0, j0 + h, h); rec(i0 + h, j0 + h, h)
        end
    end
    rec(1, 1, N)
    return leaves[], get(hist, 1, 0), hist, Ar
end

# Gradient tagging (the usual AMR refinement flag): a cell is tagged when the jump to a neighbour
# exceeds tau * range(A). Returns the number of tagged cells and the number of quadtree leaves when
# every block that contains a tagged cell is refined down to single cells.
function gradient_tags(A::AbstractMatrix, tau)
    N = size(A, 1)
    thr = tau * (maximum(A) - minimum(A))
    tag = falses(N, N)
    @inbounds for j in 1:N, i in 1:N
        g = 0.0
        i < N && (g = max(g, abs(A[i+1, j] - A[i, j])))
        i > 1 && (g = max(g, abs(A[i-1, j] - A[i, j])))
        j < N && (g = max(g, abs(A[i, j+1] - A[i, j])))
        j > 1 && (g = max(g, abs(A[i, j-1] - A[i, j])))
        tag[i, j] = g > thr
    end
    function rec(i0, j0, s)
        s == 1 && return 1
        any(@view tag[i0:i0+s-1, j0:j0+s-1]) || return 1
        h = s ÷ 2
        return rec(i0, j0, h) + rec(i0 + h, j0, h) + rec(i0, j0 + h, h) + rec(i0 + h, j0 + h, h)
    end
    return count(tag), rec(1, 1, N)
end

# Restrict to 2^n x 2^n by averaging 2x2 blocks (for comparing one field at several resolutions)
coarsen(A) = [(A[2i-1, 2j-1] + A[2i, 2j-1] + A[2i-1, 2j] + A[2i, 2j]) / 4 for i in 1:size(A, 1)÷2, j in 1:size(A, 2)÷2]

# ---------------------------------------------------------------------------------------------
# One-stop analysis of a field: rows "RANK", "QTT", "AMR", "TAG" written to io (and stdout)
# ---------------------------------------------------------------------------------------------
const EPS_RANK = [1e-2, 1e-3, 1e-4, 1e-6]
const EPS_QTT = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
const EPS_AMR = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

function writerow(io, args...)
    s = join(string.(args), " ")
    println(io, s); flush(io); println(s); flush(stdout)
end

function analyze(io, label, name, A; orderings = (:interleaved, :sequential), amr = true, qtt = true)
    n = round(Int, log2(size(A, 1)))
    rngA = maximum(A) - minimum(A)
    for ord in orderings
        T = to_tensor(A, ord)
        nrm = norm(T)
        t = @elapsed (_, spectra) = tt_svd(T; delta = 1e-13 * nrm)
        for e in EPS_RANK
            r = eps_ranks(spectra, nrm, e)
            writerow(io, "RANK", label, name, n, ord, e, maximum(r), join(r, ","))
        end
        if qtt
            for e in EPS_QTT
                tt, efro, emax = qtt_compress(A, ord, e)
                writerow(io, "QTT", label, name, n, ord, e, maxrank(tt), nparams(tt), @sprintf("%.3e", efro), @sprintf("%.3e", emax))
            end
        end
        @printf("  [%s %s %s n=%d] spectra in %.1fs\n", label, name, ord, n, t)
    end
    if amr
        for e in EPS_AMR
            leaves, nfine, hist, Ar = quadtree_bilinear(A, e * rngA)
            efro = norm(Ar - A) / norm(A); emax = maximum(abs, Ar - A) / rngA
            writerow(io, "AMR", label, name, n, e, leaves, nfine, maximum(keys(hist)), @sprintf("%.3e", efro), @sprintf("%.3e", emax))
        end
        for tau in (1e-2, 5e-2)
            ntag, leaves = gradient_tags(A, tau)
            writerow(io, "TAG", label, name, n, tau, ntag, leaves)
        end
    end
end
