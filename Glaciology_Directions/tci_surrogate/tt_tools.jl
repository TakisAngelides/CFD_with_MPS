# Tensor-train tools for the parametric surrogate. A TT is a vector of cores C_k (r_{k-1} x n_k x r_k):
# legs 1..d are the parameters (Gauss-Legendre nodes), leg d+1 is the QoI index.
#   * point evaluation on the node grid and at continuous points (barycentric Lagrange interpolation of
#     every parameter core on its GL nodes -> a functional TT),
#   * mean, variance, first-order and total Sobol indices of every QoI by exact Gauss-Legendre quadrature
#     on the TT (contractions with the weights; cost O(d n r^3)),
#   * Monte Carlo estimators (plain moments, Saltelli/Jansen pick-freeze) used for the comparison.
using LinearAlgebra, Statistics, Random

const Cores = Vector{Array{Float64,3}}

# barycentric weights for nodes t
bary_weights(t) = [1 / prod(t[i] - t[j] for j in eachindex(t) if j != i) for i in eachindex(t)]

function lagrange_row(t, lam, x)
    for i in eachindex(t)
        x == t[i] && return [j == i ? 1.0 : 0.0 for j in eachindex(t)]
    end
    v = lam ./ (x .- t)
    return v ./ sum(v)
end

# value at node indices idx (length d+1, last = QoI)
function tt_eval(C::Cores, idx::AbstractVector{<:Integer})
    v = C[1][:, idx[1], :]
    for k in 2:length(C)
        v = v * C[k][:, idx[k], :]
    end
    return v[1, 1]
end

# all QoIs at node indices of the parameters
function tt_eval_params(C::Cores, idx::AbstractVector{<:Integer})
    d = length(C) - 1
    v = C[1][:, idx[1], :]
    for k in 2:d
        v = v * C[k][:, idx[k], :]
    end
    return vec(v * C[d+1][:, :, 1])
end

# all QoIs at a continuous point u in [-1, 1]^d
function tt_eval_cont(C::Cores, t, lam, u::AbstractVector{<:Real})
    d = length(C) - 1
    v = ones(1, 1)
    for k in 1:d
        L = lagrange_row(t, lam, u[k])
        Ck = C[k]
        M = zeros(size(Ck, 1), size(Ck, 3))
        for i in eachindex(L)
            M .+= L[i] .* @view Ck[:, i, :]
        end
        v = v * M
    end
    return vec(v * C[d+1][:, :, 1])
end

# weighted mean of a core over its physical index
core_mean(Ck, w) = sum(w[i] .* Ck[:, i, :] for i in eachindex(w))

"""
    tt_stats(C, w) -> (mean, var, S1, ST)  (vectors over QoIs; S1, ST are d x nq)
Exact quadrature statistics of the TT under the product measure with weights w on every parameter leg.
"""
function tt_stats(C::Cores, w)
    d = length(C) - 1
    Cq = C[d+1][:, :, 1]                # r_d x nq
    nq = size(Cq, 2)
    m = [core_mean(C[k], w) for k in 1:d]
    # left / right products of the means
    left = Vector{Matrix{Float64}}(undef, d + 1); left[1] = ones(1, 1)
    for k in 1:d
        left[k+1] = left[k] * m[k]
    end
    right = Vector{Matrix{Float64}}(undef, d + 1); right[d+1] = Cq
    for k in d:-1:1
        right[k] = m[k] * right[k+1]
    end
    mu = vec(left[d+1] * Cq)
    # second moment: environment E (r x r) propagated with sum_i w_i C_i' E C_i
    second_moment(legs) = begin
        E = ones(1, 1)
        for k in 1:d
            Ck = C[k]
            if legs[k] === :full
                En = zeros(size(Ck, 3), size(Ck, 3))
                for i in eachindex(w)
                    Ci = Ck[:, i, :]
                    En .+= w[i] .* (Ci' * E * Ci)
                end
            else   # leg averaged before squaring
                En = m[k]' * E * m[k]
            end
            E = En
        end
        [dot(Cq[:, q], E * Cq[:, q]) for q in 1:nq]
    end
    m2 = second_moment(fill(:full, d))
    var = m2 .- mu .^ 2
    S1 = zeros(d, nq); ST = zeros(d, nq)
    for k in 1:d
        # E[f | x_k = node i] = left[k] C_k[:, i, :] right[k+1]
        g = [vec(left[k] * C[k][:, i, :] * right[k+1]) for i in eachindex(w)]    # each nq
        Vk = sum(w[i] .* g[i] .^ 2 for i in eachindex(w)) .- mu .^ 2
        S1[k, :] = Vk ./ var
        legs = fill(:full, d); legs[k] = :mean
        ST[k, :] = (m2 .- second_moment(legs)) ./ var
    end
    return mu, var, S1, ST
end

# Saltelli (2010) first-order and Jansen total-effect pick-freeze estimators. fA, fB: N x nq, fAB[k]: N x nq
function pickfreeze(fA, fB, fAB)
    d = length(fAB); nq = size(fA, 2)
    V = vec(var(vcat(fA, fB); dims = 1))
    S1 = zeros(d, nq); ST = zeros(d, nq)
    for k in 1:d
        S1[k, :] = vec(mean(fB .* (fAB[k] .- fA); dims = 1)) ./ V
        ST[k, :] = vec(mean((fA .- fAB[k]) .^ 2; dims = 1)) ./ (2V)
    end
    return S1, ST
end

# Saltelli design: rows of A, B, AB_1..AB_d (uniform on [-1, 1]^d)
function saltelli_design(rng, N::Int, d::Int)
    A = 2 .* rand(rng, N, d) .- 1; B = 2 .* rand(rng, N, d) .- 1
    AB = [begin X = copy(A); X[:, k] = B[:, k]; X end for k in 1:d]
    return A, B, AB
end

tt_ranks(C::Cores) = [size(c, 3) for c in C[1:end-1]]
tt_nparams(C::Cores) = sum(length, C)
