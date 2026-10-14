# Parameter space of the surrogate: names, ranges, nominal values, Gauss-Legendre nodes/weights and the
# map from (index or unit coordinate) to ModelParams. All parameters are independent and uniform on their
# range (Glen A log-uniform: Afac = 2^u, u uniform on [-1, 1]).
using LinearAlgebra

# name, lo, hi, nominal (used when the parameter is not active), transform of the uniform variable
const PARAM_TABLE = [
    (:log2A,  -1.0,   1.0,    0.0),      # Afac = 2^log2A  (Glen A in [0.5, 2] x 1e-16 Pa^-3 a^-1)
    (:S,      0.7e-5, 1.3e-5, 1.0e-5),   # mass-balance gradient (a^-1)
    (:R_el,   400e3,  500e3,  450e3),    # ELA radius (m)
    (:Mmax,   0.3,    0.7,    0.5),      # max accumulation (m/a)
    (:ell,    0.0,    0.3,    0.15),     # ELA ellipticity
    (:asym,   0.0,    0.4,    0.2),      # accumulation asymmetry Mmax (1 + asym cos theta)
    (:R_init, 150e3,  300e3,  225e3),    # initial radius (m) (<= smallest steady extent: advance-only start)
    (:H_init, 1000.0, 2000.0, 1500.0),   # initial dome thickness (m)
]
const PARAM_NAMES = [p[1] for p in PARAM_TABLE]
const PARAM_TEX = ["\\log_2(A/A_0)", "S", "R_{\\rm el}", "M_{\\max}", "\\epsilon", "\\alpha", "R_0", "H_0"]

# active parameter sets for d = 4, 6, 8 (the others are held at their nominal value)
active_params(d::Int) = d == 4 ? [1, 2, 3, 4] : d == 6 ? [1, 2, 3, 4, 5, 6] : d == 8 ? collect(1:8) :
                        error("d must be 4, 6 or 8")

# Gauss-Legendre nodes/weights on [-1, 1] (Golub-Welsch), weights sum to 2
function gauss_legendre(n::Int)
    b = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), b))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

struct ParamSpace
    active::Vector{Int}      # indices into PARAM_TABLE
    nnodes::Int
    t::Vector{Float64}       # GL nodes on [-1, 1]
    w::Vector{Float64}       # probability weights (sum 1)
end
function ParamSpace(d::Int, nnodes::Int)
    t, w = gauss_legendre(nnodes)
    return ParamSpace(active_params(d), nnodes, t, w ./ 2)
end
dim(ps::ParamSpace) = length(ps.active)

# unit coordinates u in [-1, 1]^d of the active parameters -> physical values of all 8 parameters
function physical(ps::ParamSpace, u::AbstractVector{<:Real})
    vals = [p[4] for p in PARAM_TABLE]
    for (k, a) in enumerate(ps.active)
        lo, hi = PARAM_TABLE[a][2], PARAM_TABLE[a][3]
        vals[a] = lo + (hi - lo) * (u[k] + 1) / 2
    end
    return vals
end
function model_params(vals::AbstractVector{<:Real})
    return ModelParams(Afac = 2.0^vals[1], S = vals[2], R_el = vals[3], Mmax = vals[4], ell = vals[5],
                       asym = vals[6], R_init = vals[7], H_init = vals[8])
end
model_params(ps::ParamSpace, u::AbstractVector{<:Real}) = model_params(physical(ps, u))
node_coords(ps::ParamSpace, idx::AbstractVector{<:Integer}) = [ps.t[i] for i in idx]
