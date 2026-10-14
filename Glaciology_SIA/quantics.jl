using ITensors
using ITensorMPS
import TensorCrossInterpolation as TCI
using TCIITensorConversion
ITensors.disable_warn_order()

# ---------------------------------------------------------------------------------------------
# d-dimensional quantics grid (d = 1, 2, 3, ...)
#
# A field u(x1_i1, ..., xd_id) on a (2^n)^d grid is stored as an MPS over d*n binary sites.
# Bit convention (same as the 1D Burgers/heat codes): bit k = 1 is the most significant bit,
# site state 1 <-> bit 0, site state 2 <-> bit 1.
#
# Ordering of the bits along the MPS chain:
#   :interleaved -> x1 y1 z1 x2 y2 z2 ... xn yn zn   (same length scale of all dims adjacent)
#   :sequential  -> x1 ... xn y1 ... yn z1 ... zn    (all x, then all y, then all z)
# Everything below only goes through grid.pos[dim][k] = MPS site of the k-th bit of dimension dim,
# so the ordering (and d) is a single switch and new orderings only need a new `pos` table.
# ---------------------------------------------------------------------------------------------

struct QuanticsGrid
    d::Int                        # number of spatial dimensions
    n::Int                        # bits per dimension -> 2^n points per dimension
    ordering::Symbol
    sites::Vector{Index{Int}}
    pos::Vector{Vector{Int}}      # pos[dim][k], k = 1 is the most significant bit
    lo::Vector{Float64}           # lower domain corner
    h::Vector{Float64}            # grid spacing per dimension
end

function QuanticsGrid(n, ordering, limits...)

    d = length(limits)
    if ordering == :interleaved
        pos = [[(k - 1) * d + a for k in 1:n] for a in 1:d]
    elseif ordering == :sequential
        pos = [[(a - 1) * n + k for k in 1:n] for a in 1:d]
    else
        error("ordering must be :interleaved or :sequential, got $(ordering)")
    end

    sites = siteinds("S=1/2", d * n)
    lo = [Float64(l[1]) for l in limits]
    h = [(l[2] - l[1]) / 2^n for l in limits]

    return QuanticsGrid(d, n, ordering, sites, pos, lo, h)

end

npoints(g::QuanticsGrid) = 2^g.n
nsites(g::QuanticsGrid) = g.d * g.n

# Storage: numbers held by an MPS/MPO, and compression factors. A state is compared with the full
# vector of N^d grid values; an operator with the nonzeros of the equivalent sparse matrix
# (stencil points per row x N^d), which is the storage an ordinary implementation would use.
nparams(psi::Union{MPS,MPO}) = sum(dim, psi)
compression(psi::MPS, g::QuanticsGrid) = 2.0^nsites(g) / nparams(psi)
compression(op::MPO, g::QuanticsGrid, stencil::Int) = stencil * 2.0^nsites(g) / nparams(op)
# from bond dimensions only (site dimension 2), for outputs that stored linkdims
nparams_from_links(links, sitedim = 2) = sum(sitedim * l * r for (l, r) in zip([1; links], [links; 1]))

# Cell-centred coordinate of 0-based index i along dimension a
coord(g::QuanticsGrid, a, i) = g.lo[a] + (i + 0.5) * g.h[a]
coords(g::QuanticsGrid, a) = [coord(g, a, i) for i in 0:npoints(g)-1]

# sigma: site states (1 or 2) of length d*n -> tuple of 0-based grid indices
function bits_to_idx(g::QuanticsGrid, sigma)

    return ntuple(g.d) do a
        i = 0
        for k in 1:g.n
            i = 2i + (sigma[g.pos[a][k]] - 1)
        end
        i
    end

end

function idx_to_bits(g::QuanticsGrid, idx)

    sigma = zeros(Int, nsites(g))
    for a in 1:g.d, k in 1:g.n
        sigma[g.pos[a][k]] = ((idx[a] >> (g.n - k)) & 1) + 1
    end

    return sigma

end

# ---------------------------------------------------------------------------------------------
# Shift operators
#
# shift_opsum(positions, +1) gives (S u)(i) = u(i+1), direction -1 gives u(i-1), acting only on
# the bits stored at `positions` (MSB first) and as identity elsewhere. This is the 1D sr1/sl1
# construction from Burgers_Equation/utilities.jl generalised to an arbitrary subset of sites,
# so the same function builds the shift along any dimension for any ordering.
# Without pbc the shift pushes zeros in from outside the domain (homogeneous Dirichlet ghost).
# ---------------------------------------------------------------------------------------------

function shift_opsum(positions, direction::Int; pbc = false)

    a, b = direction == +1 ? ("S+", "S-") : ("S-", "S+")
    m = length(positions)

    opsum = OpSum()
    for k in 1:m
        term = OpSum()
        term += a, positions[k]
        for l in k+1:m
            term *= b, positions[l]
        end
        opsum += term
    end

    if pbc
        term = OpSum()
        term += b, positions[1]
        for l in 2:m
            term *= b, positions[l]
        end
        opsum += term
    end

    return opsum

end

shift_mpo(g::QuanticsGrid, dim::Int, direction::Int; pbc = false) =
    MPO(shift_opsum(g.pos[dim], direction; pbc = pbc), g.sites)

identity_mpo(g::QuanticsGrid) = MPO(g.sites, "Id")

# Sum of MPOs with a final SVD recompression (sums of shifts have small exact rank)
# Sums are formed exactly as a direct sum of the site tensors (bond dimensions add) and then
# SVD-truncated. ITensors' default `+` is the density-matrix algorithm with a fixed cutoff of 1e-15,
# whose precision floor (~1e-8 relative) is far too coarse for residuals and operators that are
# small differences of large terms (I - dt J with dt D/h^2 ~ 1e5 at 2^16 points).
exact_sum(xs...) = length(xs) == 1 ? copy(xs[1]) : +(xs...; alg = "directsum")
mpo_lincomb(coeffs, mpos; cutoff = 1e-15, maxdim = typemax(Int)) =
    truncate!(exact_sum([c * m for (c, m) in zip(coeffs, mpos)]...); cutoff = cutoff, maxdim = maxdim)

# MPO x MPO product, "zipup" (ITensors default, truncates on the fly) or "naive" (exact, then SVD)
const MPO_PRODUCT_ALG = Ref("zipup")
mpo_product(A::MPO, B::MPO; cutoff = 1e-15, maxdim = typemax(Int)) = apply(A, B; alg = MPO_PRODUCT_ALG[], cutoff = cutoff, maxdim = maxdim)

# ---------------------------------------------------------------------------------------------
# Elementwise (Hadamard) products, sums and helpers
# ---------------------------------------------------------------------------------------------

# Diagonal MPO with the MPS entries on its diagonal (same construction as get_mpo_out_of_mps)
function diag_mpo(mps::MPS)

    mpo = MPO(length(mps))
    for (k, A) in enumerate(mps)
        s = siteind(mps, k)
        mpo[k] = A * delta(s, s', s'')
    end
    setprime!(mpo, 0; :plev => 2)

    return mpo

end

# MPO x MPS products. "naive" = exact contraction followed by SVD truncation (precision ~ eps).
# ITensors' default "densitymatrix" works with squared singular values, so its floor is
# ~ sqrt(eps) ~ 1e-8 relative to the norm of the result. That is too coarse for products of fields
# whose norms sit in different places (H^5 is dominated by the dome, |grad s|^2 by the margin):
# the floor of H^5 multiplied by the margin slopes gave O(1) errors in the margin flux at 2^12.
const PRODUCT_ALG = Ref("naive")
mps_apply(A::MPO, x::MPS; kwargs...) = apply(A, x; alg = PRODUCT_ALG[], kwargs...)

hadamard(a::MPS, b::MPS; kwargs...) = mps_apply(diag_mpo(a), b; kwargs...)

# Elementwise integer power by repeated squaring, each product truncated
function hadamard_pow(a::MPS, p::Int; kwargs...)

    p >= 1 || error("power must be >= 1")
    result = nothing
    base = a
    while p > 0
        if isodd(p)
            result = result === nothing ? base : hadamard(result, base; kwargs...)
        end
        p >>= 1
        p > 0 && (base = hadamard(base, base; kwargs...))
    end

    return result

end

# Linear combination of MPS with truncation, avoids the bond dimension growth of repeated `+`
lincomb(coeffs, mpss; kwargs...) = truncate!(exact_sum([c * m for (c, m) in zip(coeffs, mpss)]...); kwargs...)

# ||a - b|| without truncation
diff_norm(a::MPS, b::MPS) = norm(exact_sum(a, -1.0 * b))

# Product state with every entry equal to `value`
function constant_mps(g::QuanticsGrid, value)

    N = nsites(g)
    links = [Index(1, "Link,l=$(k)") for k in 1:N-1]
    tensors = ITensor[]
    for k in 1:N
        s = g.sites[k]
        inds_k = k == 1 ? (s, links[1]) : k == N ? (s, links[N-1]) : (s, links[k-1], links[k])
        c = k == 1 ? Float64(value) : 1.0
        T = ITensor(inds_k...)
        for st in 1:2
            T[s => st, [l => 1 for l in inds_k[2:end]]...] = c
        end
        push!(tensors, T)
    end

    return MPS(tensors)

end

# Sum of all entries; integral = mps_sum * prod(g.h)
mps_sum(g::QuanticsGrid, mps::MPS) = inner(constant_mps(g, 1.0), mps)

# ---------------------------------------------------------------------------------------------
# Building MPS from functions / arrays and going back to arrays
# ---------------------------------------------------------------------------------------------

# Largest |F| on a coarse lattice, used as the starting pivot (important for compact support)
function _coarse_pivot(F, g::QuanticsGrid)

    m = min(2^max(1, div(12, g.d)), npoints(g))
    coarse = round.(Int, range(0, npoints(g) - 1, length = m))
    best, bestval = nothing, -1.0
    for idx in Iterators.product(ntuple(_ -> coarse, g.d)...)
        sigma = idx_to_bits(g, idx)
        v = abs(F(sigma))
        if v > bestval
            best, bestval = sigma, v
        end
    end

    return best

end

function _tci_to_mps(F, g::QuanticsGrid; tolerance, maxbonddim, verbosity = 0, pivots = Vector{Int}[])

    tci, ranks, errors = TCI.crossinterpolate2(Float64, F, fill(2, nsites(g)), [[_coarse_pivot(F, g)]; pivots];
                                               tolerance = tolerance, maxbonddim = maxbonddim,
                                               verbosity = verbosity)

    return MPS(tci; sites = g.sites)

end

# Tensor cross interpolation of f(x1, ..., xd) directly in the chosen bit ordering
function function_to_mps(f, g::QuanticsGrid; tolerance = 1e-10, maxbonddim = 100, verbosity = 0)

    F(sigma) = begin
        idx = bits_to_idx(g, sigma)
        f(ntuple(a -> coord(g, a, idx[a]), g.d)...)
    end

    return _tci_to_mps(F, g; tolerance = tolerance, maxbonddim = maxbonddim, verbosity = verbosity)

end

# Permutation taking reshaped column-major dims [x1_lsb..x1_msb, x2_lsb..x2_msb, ...] to MPS site order
function _site_perm(g::QuanticsGrid)

    perm = zeros(Int, nsites(g))
    for a in 1:g.d, k in 1:g.n
        perm[g.pos[a][k]] = (a - 1) * g.n + (g.n - k + 1)
    end

    return perm

end

# MPS (up to cutoff) of a dense array A[i1+1, ..., id+1]; only for moderate total size
function array_to_mps(A::AbstractArray, g::QuanticsGrid; cutoff = 1e-14, maxdim = typemax(Int))

    T = permutedims(reshape(Array{Float64}(A), fill(2, nsites(g))...), _site_perm(g))

    return MPS(T, g.sites; cutoff = cutoff, maxdim = maxdim)

end

function mps_to_array(mps::MPS, g::QuanticsGrid)

    T = Array(contract(mps), g.sites...)
    A = permutedims(T, invperm(_site_perm(g)))

    return reshape(A, ntuple(_ -> npoints(g), g.d)...)

end

# Fast point evaluation with plain arrays (sampling large grids, TCI re-compression)
struct MPSEvaluator
    cores::Vector{Array{Float64,3}}   # (left link, site state, right link)
end

function MPSEvaluator(mps::MPS)

    N = length(mps)
    l = linkinds(mps)
    cores = Array{Float64,3}[]
    for k in 1:N
        s = siteind(mps, k)
        A = mps[k]
        if N == 1
            push!(cores, reshape(Array(A, s), 1, dim(s), 1))
        elseif k == 1
            push!(cores, reshape(Array(A, s, l[1]), 1, dim(s), dim(l[1])))
        elseif k == N
            push!(cores, reshape(Array(A, l[N-1], s), dim(l[N-1]), dim(s), 1))
        else
            push!(cores, Array(A, l[k-1], s, l[k]))
        end
    end

    return MPSEvaluator(cores)

end

function (E::MPSEvaluator)(sigma)

    v = E.cores[1][1, sigma[1], :]'
    for k in 2:length(E.cores)
        v = v * @view E.cores[k][:, sigma[k], :]
    end

    return v[1]

end

# Values on an m^d sub-lattice of the grid (for plotting / errors when the full grid is too big)
# Transfer of a field from grid gc (MPS Hc) to grid g by TCI of the multilinear interpolant between
# the cell centres of gc (constant beyond the outermost centres). Used for grid sequencing: a
# solution converged on a coarse grid is the initial state on a finer one (any d, any orderings).
function interpolate_mps(Hc::MPS, gc::QuanticsGrid, g::QuanticsGrid; tolerance = 1e-10, maxbonddim = 128)

    E = MPSEvaluator(Hc)
    N = npoints(gc)
    f(x...) = begin
        il = ntuple(a -> clamp(floor(Int, (x[a] - gc.lo[a]) / gc.h[a] - 0.5), 0, N - 2), gc.d)
        w = ntuple(a -> clamp((x[a] - coord(gc, a, il[a])) / gc.h[a], 0.0, 1.0), gc.d)
        val = 0.0
        for corner in Iterators.product(ntuple(_ -> (0, 1), gc.d)...)
            wt = prod(corner[a] == 1 ? w[a] : 1 - w[a] for a in 1:gc.d)
            wt == 0 && continue
            val += wt * E(idx_to_bits(gc, ntuple(a -> il[a] + corner[a], gc.d)))
        end
        val
    end

    return function_to_mps(f, g; tolerance = tolerance, maxbonddim = maxbonddim)

end

# Exact refinement of a field from grid gc to the finer grid g (same d, ordering and domain), the
# same multilinear interpolant as interpolate_mps but built from MPS operations without TCI. TCI
# reproduces the field only to its tolerance (relative), and on a fine grid that noise is amplified
# by dt D/h^2 ~ 1e8 in the residual of the first implicit step (EISMINT at 2^16 points, dt = 1000 a).
# Piecewise-constant prolongation is exact and rank-preserving: the coarse cores are kept and every
# new (finer) bit gets the core delta(l, r) [1, 1]. Multilinear interpolation is then
#   H_f = sum_{k in {-1,0,1}^d} W_k .* P(S_k Hc)
# with S_k the coarse shifts (constant extension at the domain ends, as interpolate_mps) and W_k the
# products of the 1D hat weights of the offset within the coarse cell, which depend only on the new
# bits (QTT rank <= 2 per dimension). The sum is compressed with `cutoff`.
function prolong_constant(Hc::MPS, gc::QuanticsGrid, g::QuanticsGrid)

    N = nsites(g)
    coarse_of = zeros(Int, N)                 # fine site -> coarse site (0 = new bit)
    for a in 1:g.d, k in 1:gc.n
        coarse_of[g.pos[a][k]] = gc.pos[a][k]
    end
    issorted(filter(>(0), coarse_of)) || error("coarse bits are not in the same order on the fine grid")
    tensors = ITensor[]
    cur = nothing                             # link from the last placed core to the next one
    for j in 1:N
        c = coarse_of[j]
        if c > 0
            T = replaceind(Hc[c], siteind(Hc, c), g.sites[j])
            if c > 1
                lc = linkind(Hc, c - 1)
                cur != lc && (T = replaceind(T, lc, cur))
            end
            push!(tensors, T)
            cur = c < length(Hc) ? linkind(Hc, c) : nothing
        else
            if cur === nothing
                cur = Index(1, "Link,ext")
                tensors[end] *= onehot(cur => 1)
            end
            l2 = sim(cur)
            push!(tensors, delta(cur, l2) * ITensor(ones(2), g.sites[j]))
            cur = l2
        end
    end
    cur === nothing || (tensors[end] *= onehot(cur => 1))   # trailing new bits: close the dim-1 link

    return MPS(tensors)

end

function prolong_mps(Hc::MPS, gc::QuanticsGrid, g::QuanticsGrid; cutoff = 1e-20, maxdim = typemax(Int))

    (gc.d == g.d && gc.ordering == g.ordering && g.n >= gc.n) || error("prolong_mps needs the same d and ordering and g.n >= gc.n")
    Hc = replace_siteinds(Hc, gc.sites)
    m = g.n - gc.n
    m == 0 && return replace_siteinds(Hc, g.sites)
    # coarse shifts with constant extension: S+ u(N-1) = u(N-1), S- u(0) = u(0)
    proj(a, state) = MPO(gc.sites, [k in gc.pos[a] ? state : "Id" for k in 1:nsites(gc)])
    shifts = [Dict(0 => nothing,
                   +1 => mpo_lincomb([1.0, 1.0], [shift_mpo(gc, a, +1), proj(a, "ProjDn")]; cutoff = 1e-30),
                   -1 => mpo_lincomb([1.0, 1.0], [shift_mpo(gc, a, -1), proj(a, "ProjUp")]; cutoff = 1e-30)) for a in 1:g.d]
    # hat weight of coarse neighbour k at offset o in 0:2^m-1 (xi in (-1/2, 1/2) from the coarse centre)
    hat(k, o) = (xi = (o + 0.5) / 2^m - 0.5; k == 0 ? 1 - abs(xi) : max(k * xi, 0.0))
    terms = MPS[]
    for ks in Iterators.product(ntuple(_ -> (-1, 0, 1), g.d)...)
        S = Hc
        for a in 1:g.d
            ks[a] == 0 || (S = apply(shifts[a][ks[a]], S; alg = "naive", cutoff = 1e-30))
        end
        W = function_to_mps((x...) -> prod(hat(ks[a], round(Int, (x[a] - g.lo[a]) / g.h[a] - 0.5) & (2^m - 1)) for a in 1:g.d), g;
                            tolerance = 1e-15, maxbonddim = 64)
        push!(terms, hadamard(W, prolong_constant(S, gc, g); cutoff = 1e-30))
    end

    return truncate!(exact_sum(terms...); cutoff = cutoff, maxdim = maxdim)

end

function sample_grid(mps::MPS, g::QuanticsGrid, m::Int)

    E = MPSEvaluator(mps)
    idx = round.(Int, range(0, npoints(g) - 1, length = m))
    vals = [E(idx_to_bits(g, I)) for I in Iterators.product(ntuple(_ -> idx, g.d)...)]

    return idx, vals

end

# Elementwise nonlinear map u -> h(u), re-compressed by TCI from the current MPS (e.g. max(u, 0))
function map_mps(h, mps::MPS, g::QuanticsGrid; tolerance = 1e-10, maxbonddim = 100)

    E = MPSEvaluator(mps)

    return _tci_to_mps(sigma -> h(E(sigma)), g; tolerance = tolerance, maxbonddim = maxbonddim)

end

# Elementwise map of several fields, h(u1(x), u2(x), ...)
function map_mps(h, mpss::Tuple, g::QuanticsGrid; tolerance = 1e-10, maxbonddim = 100, pivots = Vector{Int}[])

    Es = map(MPSEvaluator, mpss)

    return _tci_to_mps(sigma -> h(map(E -> E(sigma), Es)...), g; tolerance = tolerance, maxbonddim = maxbonddim, pivots = pivots)

end

# Indicator 1[f(u1, u2, ...) >= 0] of a few MPS fields by TCI, seeded with pivots at every fine cell
# around its switching points. TCI alone can miss a one-cell shift of a sharp 0/1 edge (its error
# estimate never samples the cell), which here produced a wrong active set at the ice margin and a
# Newton direction 90 % off. The switches are located on an m^d sub-lattice (m = min(N, 2^14/d))
# and, in 1D, all fine cells between two sub-lattice points of different sign become pivots; in
# d > 1 the sub-lattice points next to a switch (at most `maxpivots`).
function indicator_mps(f, mpss::Tuple, g::QuanticsGrid; tolerance = 1e-12, maxbonddim = 100, maxpivots = 4000)

    Es = map(MPSEvaluator, mpss)
    val(idx) = (sigma = idx_to_bits(g, idx); f(map(E -> E(sigma), Es)...) >= 0 ? 1.0 : 0.0)
    N = npoints(g)
    m = min(N, 2^div(14, g.d))
    sub = round.(Int, range(0, N - 1, length = m))
    pivots = Vector{Int}[]
    if g.d == 1
        v = [val((i,)) for i in sub]
        for k in 1:m-1
            v[k] == v[k+1] && continue
            for i in sub[k]:sub[k+1]
                length(pivots) < maxpivots && push!(pivots, idx_to_bits(g, (i,)))
            end
        end
    else
        V = [val(I) for I in Iterators.product(ntuple(_ -> sub, g.d)...)]
        for C in CartesianIndices(V), a in 1:g.d
            C[a] < m || continue
            D = CartesianIndex(ntuple(b -> b == a ? C[b] + 1 : C[b], g.d))
            V[C] == V[D] && continue
            for E in (C, D)
                length(pivots) < maxpivots && push!(pivots, idx_to_bits(g, ntuple(b -> sub[E[b]], g.d)))
            end
        end
    end

    return _tci_to_mps(sigma -> (f(map(E -> E(sigma), Es)...) >= 0 ? 1.0 : 0.0), g; tolerance = tolerance, maxbonddim = maxbonddim,
                       pivots = pivots)

end
