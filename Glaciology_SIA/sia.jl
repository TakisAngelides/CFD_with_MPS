# ---------------------------------------------------------------------------------------------
# Isothermal Shallow Ice Approximation (SIA) on a quantics grid, d = 1 (flowline) or 2 (map plane)
#
#   mass balance     dH/dt = a - div(q),          q = H u
#   SIA velocity     u = -Gamma H^(n+1) |grad s|^(n-1) grad s,   s = H + b
#                    Gamma = 2 A (rho g)^n / (n + 2)
#
# The velocity is computed from H and enters the mass balance through the diffusivity D = H k,
# k = Gamma H^(n+1) |grad s|^(n-1), so  div(q) = -div(D grad s).
#
# Time stepping is backward Euler with Picard linearisation, i.e. for iterate m
#
#   (I - dt L[D(H^m)]) H^(m+1) = H^old + dt a + dt L[D(H^m)] b,     L[D] = sum_a Db_a diag(D_a) Df_a
#
# where Df_a, Db_a are forward/backward differences and D_a is the diffusivity on the staggered
# faces normal to dimension a (Mahaffy type II stencil). L[D] is symmetric negative semi-definite
# for D >= 0, so I - dt L is SPD for any dt: no CFL restriction (unconditionally stable for the
# linearised step, and converged Picard = fully implicit backward Euler). Each linear system is
# solved directly in MPS form with the DMRG-like `linsolve` from ITensorMPS (KrylovKit locally),
# no MPO inverse is formed.
#
# Glen's exponent n must be an odd integer so that |grad s|^(n-1) = (|grad s|^2)^((n-1)/2) and all
# nonlinearities are polynomial, i.e. exact Hadamard products of MPS.
# ---------------------------------------------------------------------------------------------

using LinearAlgebra
using Printf

Base.@kwdef struct SIAParams
    n_glen::Int = 3
    A::Float64 = 1e-16        # Pa^-n a^-1
    rho::Float64 = 910.0      # kg m^-3
    g::Float64 = 9.81         # m s^-2
end

Gamma(p::SIAParams) = 2 * p.A * (p.rho * p.g)^p.n_glen / (p.n_glen + 2)

Base.@kwdef struct SolverSettings
    cutoff::Float64 = 1e-13          # truncation of the state H, linear-solve iterates and sums (relative)
    maxdim::Int = 128                # max bond dimension of fields
    prod_cutoff::Float64 = 1e-24     # truncation inside the nonlinear products (face fields, flux,
                                     # Jacobian coefficients). Must be tiny: H^5 is dominated by the
                                     # dome, |grad s|^2 by the margin, so a relative truncation of one
                                     # factor is amplified at the other's maximum (bench_precision.jl:
                                     # 1e-13 gives O(1) errors in the margin flux at 2^12, 1e-24 gives 1e-7)
    prod_maxdim::Int = 256           # max bond dimension inside the nonlinear products
    op_cutoff::Float64 = 1e-22       # truncation when assembling the operator MPO (relative Frobenius;
                                     # entries of I - dt J reach dt D/h^2 ~ 1e5 at 2^16 points)
    op_maxdim::Int = 128             # max bond dimension of the operator MPO
    nonlinear::Symbol = :newton      # :newton (Picard first iterate, then Newton) or :picard
    max_iters::Int = 10              # max nonlinear iterations per time step
    tol::Float64 = 1e-5              # relative change ||dH||/||H|| for nonlinear convergence
    noise_tol::Float64 = 1e-4        # accept ||dH||/||H|| below this when the line search finds no
                                     # descent: the residual is then at its truncation noise floor
                                     # (~ cond(J) sqrt(cutoff)), which grows with resolution
    line_search::Int = 6             # max step halvings in the Newton line search
    picard_first::Bool = false       # Newton: start with one Picard iterate (flattens the margin at
                                     # fine resolution, the Picard pathology, so off by default)
    extrapolate::Bool = false        # initial guess max((1+w) H^k - w H^(k-1), 0) instead of H^k (BDF2 steps)
    newton_log::Bool = false         # print every nonlinear iteration
    nsweeps::Int = 2                 # linsolve sweeps
    krylov_tol::Float64 = 1e-10      # local Krylov tolerance in linsolve
    krylov::Symbol = :gmres          # local solver in linsolve for SPD systems: :gmres or :cg
    krylov_maxiter::Int = 500        # local CG iterations
    local_solver::Symbol = :direct   # linsolve local problems: :direct (dense LU of the projected
                                     # two-site operator, exact) or :krylov (KrylovKit, see `krylov`)
    direct_maxdim::Int = 6000        # largest local problem solved directly, Krylov above
    lin_cutoff::Float64 = 1e-13      # truncation of linear-solve iterates (the Newton update only
                                     # needs ~ tol relative accuracy)
    lin_maxdim::Int = 128            # max bond dimension of linear-solve iterates (Newton updates).
                                     # Late Newton updates are mostly truncation noise of H amplified
                                     # by J, which has high rank; capping it filters the noise and
                                     # keeps the local problems small (4 lin_maxdim^2 in 1D).
    lin_res_tol::Float64 = 2e-2      # re-sweep linsolve while ||rhs - A x||/||rhs|| is above this. Good
                                     # solves reach ~3e-3 (floor set by truncation noise amplified by A),
                                     # failed local solves show up as O(0.1-10)
    lin_retries::Int = 3             # max extra linsolve rounds (each with 2x the sweeps), best x kept
    lin_tol::Float64 = 0.0           # > 0: sweep linsolve until the exact relative residual ||rhs - A x||/||rhs||
                                     # is below lin_tol (checked after every sweep), at most lin_maxsweeps sweeps;
                                     # 0: fixed nsweeps (+ re-sweeps above lin_res_tol)
    lin_maxsweeps::Int = 64
    lin_refine::Int = 0              # iterative refinement rounds: x += linsolve(A, rhs - A x) (exact residual),
                                     # while the residual is above lin_refine_tol (fine 1D grids: the DMRG solve
                                     # stalls at ~1e-2 at 2^18 although its correction solves still make progress)
    lin_refine_tol::Float64 = 1e-6
    linear_solver::Symbol = :linsolve  # :linsolve (DMRG-like MPS solve) or :inverse (tracked MPO inverse)
    inv_cutoff::Float64 = 1e-14      # truncation of the Newton-Schulz products
    inv_maxdim::Int = 256            # max bond dimension of the inverse MPO
    inv_maxiter::Int = 60            # max Newton-Schulz iterations per solve
    solve_tol::Float64 = 1e-6        # relative residual ||rhs - A x|| / ||rhs|| for the inverse solver
    positivity::Bool = false         # project H <- max(H, 0) after each step (TCI re-compression)
    complementarity::Bool = false    # with a mass balance: solve the complementarity problem H >= 0,
                                     # R(H) >= 0, H R(H) = 0 by semismooth Newton on min(H, R(H))
                                     # (ice-free cells where the step would melt more than is there)
    ncp_tol::Float64 = 1e-12         # TCI tolerance of min(H, R) and of the active-set mask
    ncp_exact_mask::Bool = false     # debug (small grids): form the mask from dense arrays instead of TCI
    ncp_mask_alpha::Float64 = 0.0    # > 0: smooth mask m = 1 - exp(-alpha (q + |q|)), q = H - c R (alpha in 1/m),
                                     # TCI of a continuous surrogate of the indicator (Peddinti et al. 2024,
                                     # Commun. Phys. 7, 135); 0: sharp 0/1 mask, TCI seeded at the switches
    ncp_scale::Float64 = 1.0         # c in min(H, c R): same solutions, different active-set guess. R ~ dt x
                                     # (net ablation), so with c = 1 a margin cell of thickness H is guessed
                                     # ice-free as soon as dt |a + F| > H, and Newton tries to delete it at once
    limit_ablation::Bool = false     # (simpler, dt-dependent alternative) mass balance limited by the available ice:
                                     # rhs = max(history + dt a, 0), so ice-free ablation zones stay at
                                     # H = 0 instead of H = dt a < 0 (negative D ~ H^5 = anti-diffusion)
    positivity_tol::Float64 = 1e-8
    verbose::Bool = true
end

# Constant finite-difference MPOs, built once per grid
struct SIAOperators
    I::MPO
    Df::Vector{MPO}                  # (S+ - I)/h            gradient on face a+1/2
    Db::Vector{MPO}                  # (I - S-)/h            divergence back to cells
    Avg::Vector{MPO}                 # (I + S+)/2            cell -> face a+1/2
    Gface::Matrix{MPO}               # Gface[a, b]: d/dx_b on face a+1/2 (a == b: Df_a)
    Gcell::Vector{MPO}               # (S+ - S-)/(2h)        central gradient at cells
end

function SIAOperators(g::QuanticsGrid)

    I = identity_mpo(g)
    Sp = [shift_mpo(g, a, +1) for a in 1:g.d]
    Sm = [shift_mpo(g, a, -1) for a in 1:g.d]

    Df = [mpo_lincomb([1 / g.h[a], -1 / g.h[a]], [Sp[a], I]) for a in 1:g.d]
    Db = [mpo_lincomb([1 / g.h[a], -1 / g.h[a]], [I, Sm[a]]) for a in 1:g.d]
    Avg = [mpo_lincomb([0.5, 0.5], [I, Sp[a]]) for a in 1:g.d]
    Gcell = [mpo_lincomb([1 / (2g.h[a]), -1 / (2g.h[a])], [Sp[a], Sm[a]]) for a in 1:g.d]
    Gface = [a == b ? Df[a] : mpo_product(Avg[a], Gcell[b]) for a in 1:g.d, b in 1:g.d]

    return SIAOperators(I, Df, Db, Avg, Gface, Gcell)

end

# Truncation inside the nonlinear products
prod_kw(st) = (; cutoff = st.prod_cutoff, maxdim = st.prod_maxdim)

# |grad s|^2 from its components
grad_sq(grads::Vector{MPS}, st::SolverSettings) =
    lincomb(ones(length(grads)), [hadamard(gc, gc; prod_kw(st)...) for gc in grads]; prod_kw(st)...)

# base .* H^p by p successive Hadamard products, starting from the gradient factors. The order
# matters for precision: H^p alone is dominated by the dome while the gradients are largest at the
# margin, so forming H^5 first and truncating it relative to its (dome) norm destroys its values in
# the thin margin cells (H^5 there is ~1e-10 of the dome value at 2^16 points), which are exactly
# where the operator is stiffest. Every intermediate H^k |grad s|^2 is balanced instead.
function mul_pow(base::MPS, H::MPS, p::Int; kwargs...)
    for _ in 1:p
        base = hadamard(H, base; kwargs...)
    end
    return base
end

# Velocity coefficient k = Gamma H^(n+1) |grad s|^(n-1), so that u = -k grad s
function velocity_coefficient(H::MPS, g2, p::SIAParams, st::SolverSettings)

    kw = prod_kw(st)
    if p.n_glen > 1
        k = mul_pow(hadamard_pow(g2, div(p.n_glen - 1, 2); kw...), H, p.n_glen + 1; kw...)
    else
        k = hadamard_pow(H, p.n_glen + 1; kw...)
    end

    return Gamma(p) * k

end

# Fields on the staggered faces normal to dimension a (Mahaffy type II stencil)
struct FaceFields
    H::MPS                 # face thickness (I + S_a+)/2 H
    grads::Vector{MPS}     # grads[c] = ds/dx_c on the face
    g2::Union{MPS,Nothing} # |grad s|^2 (nothing for n = 1)
    k::MPS                 # velocity coefficient, face velocity u_a = -k grads[a]
    D::MPS                 # diffusivity H k, flux q_a = -D grads[a]
end

function face_fields(H::MPS, b, ops::SIAOperators, p::SIAParams, st::SolverSettings)

    kw = prod_kw(st)
    s = b === nothing ? H : lincomb([1.0, 1.0], [H, b]; kw...)
    d = length(ops.Df)

    F = FaceFields[]
    for a in 1:d
        Ha = mps_apply(ops.Avg[a], H; kw...)
        grads = [mps_apply(ops.Gface[a, c], s; kw...) for c in 1:d]
        g2 = p.n_glen > 1 ? grad_sq(grads, st) : nothing
        ka = velocity_coefficient(Ha, g2, p, st)
        push!(F, FaceFields(Ha, grads, g2, ka, hadamard(Ha, ka; kw...)))
    end

    return F

end

face_diffusivities(H::MPS, b, ops::SIAOperators, p::SIAParams, st::SolverSettings) =
    [f.D for f in face_fields(H, b, ops, p, st)]

# sum_a Db_a diag(c_a) O_a as one MPO
function sum_db_diag_op(coeffs::Vector{MPS}, opsr::Vector{MPO}, dims::Vector{Int}, ops::SIAOperators, st::SolverSettings)

    kw = (; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
    terms = [mpo_product(ops.Db[a], mpo_product(diag_mpo(c), O; kw...); kw...) for (c, O, a) in zip(coeffs, opsr, dims)]

    return length(terms) == 1 ? terms[1] : truncate!(exact_sum(terms...); kw...)

end

# L[D] = sum_a Db_a diag(D_a) Df_a (symmetric, negative semi-definite for D >= 0)
diffusion_mpo(D::Vector{MPS}, ops::SIAOperators, st::SolverSettings) =
    sum_db_diag_op(D, ops.Df, collect(eachindex(D)), ops, st)

# Newton Jacobian of div(D(H) grad s) with respect to H, as one MPO. With
#   D_a = Gamma H_a^(n+2) g2^m,  m = (n-1)/2,  P_a = Gamma H_a^(n+2) g2^(m-1)
#   d(D_a g_a) = D_a dg_a + (n+2) k_a g_a dH_a + (n-1) P_a g_a sum_c g_c dg_c
# with dH_a = Avg_a dH and dg_c = Gface[a, c] dH. Not symmetric (advective part).
function jacobian_mpo(F::Vector{FaceFields}, ops::SIAOperators, p::SIAParams, st::SolverSettings)

    kw = prod_kw(st)
    n = p.n_glen
    d = length(F)

    coeffs = MPS[]; opsr = MPO[]; dims = Int[]
    for a in 1:d
        f = F[a]
        ga = f.grads[a]
        # (n-1) P g_a g_c = (n-1) Gamma H^(n+2) g2^(m-1) g_a g_c, formed gradients first (see mul_pow)
        Pterm(gc) = begin
            q = hadamard(ga, gc; kw...)
            n > 3 && (q = hadamard(q, hadamard_pow(f.g2, div(n - 3, 2); kw...); kw...))
            (n - 1) * Gamma(p) * mul_pow(q, f.H, n + 2; kw...)
        end
        # dg_a terms (diffusive part)
        push!(coeffs, n > 1 ? lincomb([1.0, 1.0], [f.D, Pterm(ga)]; kw...) : f.D)
        push!(opsr, ops.Df[a]); push!(dims, a)
        # dH_a term (advective part from the H dependence of D)
        push!(coeffs, (n + 2) * hadamard(f.k, ga; kw...)); push!(opsr, ops.Avg[a]); push!(dims, a)
        # transverse gradient terms
        if n > 1
            for c in 1:d
                c == a && continue
                push!(coeffs, Pterm(f.grads[c])); push!(opsr, ops.Gface[a, c]); push!(dims, a)
            end
        end
    end

    return sum_db_diag_op(coeffs, opsr, dims, ops, st)

end

# div(D grad s) = sum_a Db_a (D_a g_a) as an MPS (= -div q)
function flux_divergence(F::Vector{FaceFields}, ops::SIAOperators, st::SolverSettings)

    kw = prod_kw(st)
    parts = [mps_apply(ops.Db[a], hadamard(F[a].D, F[a].grads[a]; kw...); kw...) for a in eachindex(F)]

    return length(parts) == 1 ? parts[1] : lincomb(ones(length(parts)), parts; kw...)

end

# Cell-centred velocity components u_a = -Gamma H^(n+1) |grad s|^(n-1) ds/dx_a (diagnostic output)
function velocity(H::MPS, b, ops::SIAOperators, p::SIAParams, st::SolverSettings)

    kw = prod_kw(st)
    s = b === nothing ? H : lincomb([1.0, 1.0], [H, b]; kw...)
    grads = [mps_apply(G, s; kw...) for G in ops.Gcell]
    k = velocity_coefficient(H, p.n_glen > 1 ? grad_sq(grads, st) : nothing, p, st)

    return [-1.0 * hadamard(k, gc; kw...) for gc in grads]

end

# ||b - A x|| / ||b||. A x is formed by an exact contraction, b - A x as an exact direct sum, and the
# sum is then brought to orthogonal form (SVD sweep, no truncation beyond 1e-30) before taking the
# norm: orthogonalization resolves the cancellation between b and A x with absolute accuracy
# ~ eps ||b||. Expanding <b,b> - 2<b|A|x> + <Ax|Ax> instead loses everything once sigma_max(A)
# ~ 1e6-1e7 (2^16-2^18 points): it reported residuals of O(1) for solutions accurate to 1e-4.
# When the exact product would be large (chi(A) chi(x) > 2000, 2D) the expanded form is used; it is
# adequate there because the 2D grids in use are far less ill-conditioned.
function residual_norm(A::MPO, x::MPS, b::MPS)
    if maxlinkdim(A) * maxlinkdim(x) > 2000
        bb = inner(b, b)
        r2 = bb - 2 * real(inner(b', A, x)) + real(inner(A, x, A, x))
        return sqrt(max(r2, 0.0) / bb)
    end
    Ax = mps_apply(A, x; cutoff = 1e-30, maxdim = typemax(Int))
    r = lincomb([1.0, -1.0], [b, Ax]; cutoff = 1e-30)
    return norm(r) / norm(b)
end

# Local updater for linsolve that assembles the projected two-site operator (environments x MPO
# tensors) as a dense matrix and solves it by LU. The local problems inherit the conditioning of
# the global one (cond ~ 4^nbits), which is what made the unpreconditioned local GMRES stall or
# break down at 2^12 points and beyond; an exact local solve removes the dependence on the local
# condition number. In 1D the local dimension is 4 chi^2 ~ 400-4000, so this is cheap; above
# `maxdirect` it falls back to KrylovKit.
function direct_updater(problem, init; internal_kwargs, coefficients, maxdirect = 6000, krylov_kwargs = (;), kwargs...)
    P = problem.reduced_operator
    b = dag(only([contract(c) for c in problem.reduced_constant_terms]))
    a0, a1 = coefficients
    if dim(init) > maxdirect
        x, info = linsolve(P, b, init, a0, a1; krylov_kwargs...)
        return x, (; info)
    end
    op = foldl(*, P.H[(P.lpos + 1):(P.rpos - 1)])
    P.lpos >= 1 && (op = P.LR[P.lpos] * op)
    P.rpos <= length(P.H) && (op = op * P.LR[P.rpos])
    C = combiner(inds(init)...)
    c = combinedind(C)
    M = matrix(op * C * prime(C), prime(c), c)
    a1 != 1 && (M .*= a1)
    a0 != 0 && (M += a0 * LinearAlgebra.I)
    rhs = vector(b * C)
    F = lu(M; check = false)
    xv = issuccess(F) ? F \ rhs : qr(M, ColumnNorm()) \ rhs
    return itensor(xv, c) * dag(C), (; info = nothing)
end

# One linsolve call with the configured local solver
function _linsolve_once(A, rhs, x0, st, spd, nsweeps; checkdone = (; kw...) -> false)
    krylov_kw = spd && st.krylov == :cg ? (; ishermitian = true, isposdef = true, tol = st.krylov_tol, maxiter = st.krylov_maxiter) :
                                          (; ishermitian = spd, tol = st.krylov_tol, maxiter = 30, krylovdim = 30)
    if st.local_solver == :direct
        return linsolve(A, rhs, x0; nsweeps = nsweeps, cutoff = st.lin_cutoff, maxdim = st.lin_maxdim, outputlevel = 0, checkdone,
                        updater = direct_updater, updater_kwargs = (; maxdirect = st.direct_maxdim, krylov_kwargs = krylov_kw))
    else
        return linsolve(A, rhs, x0; nsweeps = nsweeps, cutoff = st.lin_cutoff, maxdim = st.lin_maxdim, outputlevel = 0, checkdone, updater_kwargs = krylov_kw)
    end
end

# Linear solve with an exact residual check: re-sweep (2x the sweeps each round) while the
# relative residual is above lin_res_tol, keep the best iterate.
function _linsolve(A, rhs, x0, st, spd)
    if st.lin_tol > 0
        # tolerance-controlled: stop after the first sweep whose exact residual is below lin_tol
        nsw = Ref(0)
        done = (; state, sweep, kw...) -> (nsw[] = sweep; residual_norm(A, state, rhs) < st.lin_tol)
        x = _linsolve_once(A, rhs, x0, st, spd, st.lin_maxsweeps; checkdone = done)
        return x, (residual_norm(A, x, rhs), nsw[])
    end
    x = _linsolve_once(A, rhs, x0, st, spd, st.nsweeps)
    res = residual_norm(A, x, rhs)
    tries = 0
    while res > st.lin_res_tol && tries < st.lin_retries
        tries += 1
        x_try = _linsolve_once(A, rhs, x, st, spd, 2^tries * st.nsweeps)
        res_try = residual_norm(A, x_try, rhs)
        res_try < res && ((x, res) = (x_try, res_try))
    end
    for _ in 1:st.lin_refine
        res <= st.lin_refine_tol && break
        r = lincomb([1.0, -1.0], [rhs, mps_apply(A, x; cutoff = 1e-30, maxdim = typemax(Int))]; cutoff = st.lin_cutoff, maxdim = 4 * st.lin_maxdim)
        dx = _linsolve_once(A, r, r, st, spd, st.nsweeps)
        x_try = lincomb([1.0, 1.0], [x, dx]; cutoff = st.lin_cutoff, maxdim = st.lin_maxdim)
        res_try = residual_norm(A, x_try, rhs)
        res_try < res || break
        x, res = x_try, res_try
        tries += 1
    end
    return x, (res, tries)
end

# ---------------------------------------------------------------------------------------------
# Tracked approximate inverse (Newton-Schulz), the 2D/high-resolution version of the MPO-inverse
# idea of Burgers_Equation/mpo_inverse.jl:
#   X <- X + X (I - A X)       (error R = I - A X is squared every iteration)
# From scratch (X0 = I / lambda_max) it needs ~ log2(cond(A)) iterations, but A changes little
# between Picard iterations and time steps, so warm-starting from the previous X needs only a few.
# Each linear solve is then x = X rhs plus iterative refinement x <- x + X (rhs - A x), all plain
# MPO-MPS products, and its cost does not depend on the conditioning once X is converged.
# The stopping test uses the residual of the actual solve: the Frobenius norm of I - A X is
# dominated by the (quickly converging) high-frequency modes and would hide slow smooth modes.
# ---------------------------------------------------------------------------------------------

mutable struct InverseTracker
    X::Union{MPO,Nothing}
    total_iters::Int
end
InverseTracker() = InverseTracker(nothing, 0)

function solve_with_inverse!(tr::InverseTracker, A::MPO, rhs::MPS, lambda_max_fn, ops::SIAOperators, st::SolverSettings)

    kw_op = (; cutoff = st.inv_cutoff, maxdim = st.inv_maxdim)
    kw = (; cutoff = st.cutoff, maxdim = st.maxdim)
    tr.X === nothing && (tr.X = (1 / lambda_max_fn()) * ops.I)
    nrhs = norm(rhs)

    # x = X rhs + two refinement steps; returns (x, relative residual)
    refined(X) = begin
        x = apply(X, rhs; kw...)
        res = Inf
        for _ in 1:2
            res = residual_norm(A, x, rhs)
            res < st.solve_tol && break
            r = lincomb([1.0, -1.0], [rhs, apply(A, x; kw...)]; kw...)
            x = lincomb([1.0, 1.0], [x, apply(X, r; kw...)]; kw...)
        end
        res = residual_norm(A, x, rhs)
        x, res
    end

    # Newton-Schulz updates of X until the solve residual stops improving. Once X is at its bond
    # dimension cap further iterations amplify truncation errors, so stagnation ends the loop and
    # the best (x, X) is kept.
    x, res = refined(tr.X)
    ns = 0
    for k in 1:st.inv_maxiter
        res < st.solve_tol && break
        R = mpo_lincomb([1.0, -1.0], [ops.I, apply(A, tr.X; kw_op...)]; kw_op...)
        X_new = mpo_lincomb([1.0, 1.0], [tr.X, apply(tr.X, R; kw_op...)]; kw_op...)
        x_new, res_new = refined(X_new)
        ns += 1
        # accept improvements; near the noise floor (res < 0.1) stop once the gain is < 2x
        res_new < res || break
        stagnating = res < 0.1 && res_new > 0.5 * res
        tr.X, x, res = X_new, x_new, res_new
        stagnating && break
    end
    tr.total_iters += ns

    return x, (; ns_iters = ns, residual = res, X_maxdim = maxlinkdim(tr.X))

end

# Gershgorin bound for lambda_max(I - dt L[D]) from sampled face diffusivities
function lambda_max_estimate(D::Vector{MPS}, dt, g::QuanticsGrid)
    Dmax = [maximum(abs, sample_grid(Da, g, min(npoints(g), 2^max(4, div(16, g.d))))[2]) for Da in D]
    return 1.5 * (1 + dt * sum(4 * Dmax[a] / g.h[a]^2 for a in eachindex(D)))
end

# One implicit step of size dt (years), returns the new H and a NamedTuple of diagnostics
#   H_prev === nothing -> backward Euler (BDF1):  H - dt F(H) = H_old + dt a
#   H_prev = H^(n-1)   -> BDF2:                   H - (2/3) dt F(H) = (4 H_old - H_prev)/3 + (2/3) dt a
# with F(H) = div(D(H) grad s). Both are A-stable (BDF1 also L-stable). The nonlinear system is
# solved by Picard (lagged diffusivity, SPD linear systems) or Newton (first iterate Picard, then
# full Jacobian, non-symmetric linear systems), each linear system with MPS linsolve.
function implicit_step(H_old::MPS, H_prev, dt, a_mb, b, g::QuanticsGrid, ops::SIAOperators, p::SIAParams, st::SolverSettings;
                       tracker::Union{InverseTracker,Nothing} = nothing, omega = 1.0)

    kw = (; cutoff = st.cutoff, maxdim = st.maxdim)

    # History part of the right-hand side and the effective implicit time step. Variable-step BDF2
    # with omega = dt_n / dt_(n-1):
    #   H^(n+1) - dt (1+w)/(1+2w) F(H^(n+1)) = [(1+w)^2 H^n - w^2 H^(n-1)] / (1+2w)
    # (omega = 1: (4 H^n - H^(n-1))/3 and 2dt/3)
    if H_prev === nothing
        hist, dt_eff = H_old, dt
    else
        w = omega
        hist = lincomb([(1 + w)^2 / (1 + 2w), -w^2 / (1 + 2w)], [H_old, H_prev]; kw...)
        dt_eff = dt * (1 + w) / (1 + 2w)
    end
    if a_mb === nothing
        rhs0 = hist
    elseif st.limit_ablation
        rhs0 = map_mps((h, a) -> max(h + dt_eff * a, 0.0), (hist, a_mb), g; tolerance = 1e-10, maxbonddim = st.maxdim)
    else
        rhs0 = lincomb([1.0, dt_eff], [hist, a_mb]; kw...)
    end
    dt = dt_eff

    # initial guess: H^n, or the linear extrapolation max((1+w) H^n - w H^(n-1), 0), which puts a
    # moving margin approximately where it will be (Newton advances a degenerate front only about
    # one cell per iteration)
    H = if st.extrapolate && H_prev !== nothing
        map_mps((h1, h0) -> max((1 + omega) * h1 - omega * h0, 0.0), (H_old, H_prev), g; tolerance = 1e-12, maxbonddim = st.maxdim)
    else
        H_old
    end
    change = Inf
    op_dim = 0; op_np = 0
    it = 0
    t_D = t_op = t_solve = 0.0
    ns_iters = 0; inv_res = NaN; X_dim = 0; lin_res = (NaN, 0)
    ls_failed = false
    noise_accept = false
    for m in 1:st.max_iters
        it = m

        t_D += @elapsed F = face_fields(H, b, ops, p, st)

        if st.nonlinear == :newton && (m > 1 || !st.picard_first)
            # J dH = -R,  R = H - dt div(D grad s) - rhs0,  J = I - dt dF/dH, with backtracking line
            # search on ||R||. Picard alone does not converge at fine resolution: the margin
            # diffusivity (~1e10 m^2/a at 600 m spacing) flattens the margin in one frozen-D solve,
            # D ~ |grad s|^2 then vanishes there and the iteration oscillates (margin and dome stall).
            residual(Hc, Fc) = lincomb([1.0, -dt, -1.0], [Hc, flux_divergence(Fc, ops, st), rhs0]; kw...)
            # With a mass balance, ice-free cells must stay at H = 0 where the step would ablate more
            # than is there: complementarity H >= 0, R >= 0, H R = 0, written as Phi = min(H, R) = 0.
            # Semismooth Newton: rows in the active set (H < R, i.e. H = 0 is enforced) become
            # identity rows, J = I - dt diag(mask) dF/dH with mask = 1 where the flow equation holds.
            # min and mask are elementwise maps of two MPS, formed by TCI (a 0/1 mask with a few
            # switching points has QTT rank <= 2 per switch in 1D).
            # Only the mask is formed by TCI: it is a clean 0/1 step function (in ice-free ablation
            # cells R ~ dt |a| >> 0, under the ice H >> R). Phi itself is then formed exactly as
            # mask .* R + (1 - mask) .* H. Forming min(H, R) by TCI failed near steady state, where Phi
            # is noise-like everywhere and its TCI is inaccurate (noisy merit, failing line searches).
            ncp = st.complementarity && a_mb !== nothing
            c = st.ncp_scale
            active_mask(Hc, Rc) = st.ncp_exact_mask ?
                array_to_mps(Float64.(mps_to_array(Hc, g) .>= c .* mps_to_array(Rc, g)), g; cutoff = 1e-30) :
                st.ncp_mask_alpha > 0 ?
                map_mps((h, r) -> (q = h - c * r; 1 - exp(-st.ncp_mask_alpha * (q + abs(q)))), (Hc, Rc), g; tolerance = st.ncp_tol, maxbonddim = st.maxdim) :
                indicator_mps((h, r) -> h - c * r, (Hc, Rc), g; tolerance = st.ncp_tol, maxbonddim = st.maxdim)
            phi(Hc, Rc, mk) = lincomb([c, 1.0, -1.0], [hadamard(mk, Rc; prod_kw(st)...), Hc, hadamard(mk, Hc; prod_kw(st)...)]; kw...)
            merit(Hc, Rc) = ncp ? phi(Hc, Rc, active_mask(Hc, Rc)) : Rc
            t_op += @elapsed begin
                R = residual(H, F)
                mask = ncp ? active_mask(H, R) : nothing
                Phi = ncp ? phi(H, R, mask) : R
                nR = norm(Phi)
                Jac = jacobian_mpo(F, ops, p, st)
                if ncp
                    # d Phi/dH = (1 - m) + c m (I - dt dF/dH) = I + (c - 1) diag(m) - c dt diag(m) dF/dH
                    Jac = mpo_product(diag_mpo(mask), Jac; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
                    J = c == 1 ? mpo_lincomb([1.0, -dt], [ops.I, Jac]; cutoff = st.op_cutoff, maxdim = st.op_maxdim) :
                                 mpo_lincomb([1.0, c - 1, -c * dt], [ops.I, diag_mpo(mask), Jac]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
                else
                    J = mpo_lincomb([1.0, -dt], [ops.I, Jac]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
                end
            end
            op_dim = maxlinkdim(J); op_np = nparams(J)
            t_solve += @elapsed dH, lin_res = _linsolve(J, -1.0 * Phi, -1.0 * Phi, st, false)
            # convergence is judged on the full Newton correction (the error estimate of the
            # iterate), not on the damped step, which a failing line search makes small
            newton_change = norm(dH) / max(norm(H), eps())
            lam = 1.0
            armijo = false
            H_new = lincomb([1.0, lam], [H, dH]; kw...)
            for _ in 0:st.line_search
                t_D += @elapsed nR_new = norm(merit(H_new, residual(H_new, face_fields(H_new, b, ops, p, st))))
                armijo = nR_new <= (1 - 1e-4 * lam) * nR
                armijo && break
                lam /= 2
                H_new = lincomb([1.0, lam], [H, dH]; kw...)
            end
            st.newton_log && println(@sprintf("    newton %2d: |R|/|rhs| %.2e  |dH|/|H| %.2e  lambda %s  lin res %.1e (%d)  chi(J) %d chi(dH) %d",
                                              m, nR / norm(rhs0), newton_change, armijo ? @sprintf("%.3g", lam) : "fail",
                                              lin_res[1], lin_res[2], op_dim, maxlinkdim(dH)))
            st.newton_log && flush(stdout)
            if !armijo
                # no descent along the Newton direction: either the residual is at its truncation
                # noise floor (then the correction is small and H is converged), or the step is not
                # solvable from here (margin would have to move many cells) -> report failure
                change = newton_change
                ls_failed = newton_change >= st.noise_tol
                if !ls_failed
                    # at the noise floor: take the (small) full Newton step rather than keeping H,
                    # which would silently freeze a slowly evolving state
                    H = lincomb([1.0, 1.0], [H, dH]; kw...)
                    noise_accept = true
                    change = 0.0
                end
                break
            end
            change = newton_change
            H = H_new
            change < st.tol && break
            continue
        else
            # (I - dt L[D(H)]) H_new = rhs0 + dt L[D(H)] b
            t_op += @elapsed begin
                L = diffusion_mpo([f.D for f in F], ops, st)
                A = mpo_lincomb([1.0, -dt], [ops.I, L]; cutoff = st.op_cutoff, maxdim = st.op_maxdim)
            end
            op_dim = maxlinkdim(A); op_np = nparams(A)
            rhs = b === nothing ? rhs0 : lincomb([1.0, dt], [rhs0, mps_apply(L, b; kw...)]; kw...)
            if st.linear_solver == :inverse
                t_solve += @elapsed H_new, sinfo = solve_with_inverse!(tracker, A, rhs, () -> lambda_max_estimate([f.D for f in F], dt, g), ops, st)
                ns_iters += sinfo.ns_iters; inv_res = sinfo.residual; X_dim = sinfo.X_maxdim; lin_res = (inv_res, 0)
            else
                t_solve += @elapsed H_new, lin_res = _linsolve(A, rhs, H, st, true)
            end
        end

        change = diff_norm(H_new, H) / max(norm(H), eps())
        H = H_new
        change < st.tol && break

    end

    converged = change < st.tol && !ls_failed && isfinite(norm(H))
    if st.positivity && converged
        H = map_mps(x -> max(x, 0.0), H, g; tolerance = st.positivity_tol, maxbonddim = st.maxdim)
    end

    return H, (; converged = converged, noise_accept = noise_accept, iters = it, change = change, op_maxdim = op_dim, op_params = op_np, H_maxdim = maxlinkdim(H), H_params = nparams(H),
                t_D = t_D, t_op = t_op, t_solve = t_solve, ns_iters = ns_iters, inv_res = inv_res, X_maxdim = X_dim, lin_res = lin_res)

end

# Largest diffusivity, for reporting how far beyond the explicit stability limit dt is
function explicit_dt_limit(H::MPS, b, g::QuanticsGrid, ops::SIAOperators, p::SIAParams, st::SolverSettings)

    D = face_diffusivities(H, b, ops, p, st)
    Dmax = maximum(maximum(abs, sample_grid(Da, g, min(npoints(g), 256))[2]) for Da in D)

    return minimum(g.h)^2 / (2 * g.d * Dmax), Dmax

end
