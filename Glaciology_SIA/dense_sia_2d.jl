# Dense (sparse-matrix) 2D EISMINT I moving margin to steady state, Mahaffy type II stencil as sia.jl,
# to test a smooth margin regularisation before building it into the MPS solver:
#   heps > 0   ablation limited by the available ice, a_eff = a (1 - exp(-H/heps)) where a < 0, so the
#              equation is smooth everywhere (no complementarity) and the margin ends in a thin tail
#   heps = 0   semismooth Newton on min(H, R) (complementarity, sharp margin) as in sia.jl
# Backward Euler with growing dt to steady state; Newton Jacobian exact by complex step (the 3x3
# stencil needs 9 colours). Grid sequencing: init = h5 file of a coarser run (bilinear). At the end the
# QTT ranks (interleaved, SVD at several cutoffs) of the steady H and of the x-face diffusivity.
#   sbatch --time=02:00:00 slurm/run.sbatch dense_sia_2d.jl nbits=7 heps=1
using LinearAlgebra, SparseArrays, Printf, HDF5
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")

cfg = Dict{String,Any}("nbits" => 7, "heps" => 1.0, "dt0" => 100.0, "dt_max" => 1e5, "t_max" => 3e5, "tol" => 1e-9, "max_iters" => 12,
                       "init" => "", "tag" => "", "steady_tol" => 1e-10, "cutoffs" => "1e-8,1e-12,1e-16")
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end
p = SIAParams(); eb = EISMINTMovingMargin(p = p); L = eb.L; G = Gamma(p); n = p.n_glen
nb = cfg["nbits"]; N = 2^nb; h = 2L / N; heps = cfg["heps"]
xs = [-L + (i - 0.5) * h for i in 1:N]
a = [eismint_mb(eb, hypot(x, y)) for x in xs, y in xs]

pad(H) = (P = zeros(eltype(H), N + 2, N + 2); P[2:N+1, 2:N+1] .= H; P)
# face diffusivities and fluxes (zero ghost cells), F = div(D grad H) (flat bed, s = H)
function fluxdiv(H)
    P = pad(H); T = eltype(H)
    Fx = zeros(T, N + 1, N); Fy = zeros(T, N, N + 1)          # fluxes D ds on faces i-1/2 (i = 1..N+1)
    for j in 1:N, i in 0:N
        I, J = i + 1, j + 1
        Hf = (P[I, J] + P[I+1, J]) / 2; sx = (P[I+1, J] - P[I, J]) / h
        sy = (P[I, J+1] - P[I, J-1] + P[I+1, J+1] - P[I+1, J-1]) / (4h)
        Fx[i+1, j] = G * Hf^(n + 2) * (sx^2 + sy^2)^div(n - 1, 2) * sx
    end
    for j in 0:N, i in 1:N
        I, J = i + 1, j + 1
        Hf = (P[I, J] + P[I, J+1]) / 2; sy = (P[I, J+1] - P[I, J]) / h
        sx = (P[I+1, J] - P[I-1, J] + P[I+1, J+1] - P[I-1, J+1]) / (4h)
        Fy[i, j+1] = G * Hf^(n + 2) * (sx^2 + sy^2)^div(n - 1, 2) * sy
    end
    return (Fx[2:end, :] - Fx[1:end-1, :]) / h + (Fy[:, 2:end] - Fy[:, 1:end-1]) / h
end
function Dx(H)
    P = pad(H); D = zeros(N, N)
    for j in 1:N, i in 1:N
        I, J = i + 1, j + 1
        Hf = (P[I, J] + P[I+1, J]) / 2; sx = (P[I+1, J] - P[I, J]) / h
        sy = (P[I, J+1] - P[I, J-1] + P[I+1, J+1] - P[I+1, J-1]) / (4h)
        D[i, j] = G * Hf^(n + 2) * (sx^2 + sy^2)^div(n - 1, 2)
    end
    return D
end
a_eff(H) = heps > 0 ? ifelse.(a .< 0, a .* (1 .- exp.(-H ./ heps)), a) : a
# residual of a backward Euler step and its exact Jacobian (complex step, 9 colours)
Res(H, Hold, dt) = H - dt * (fluxdiv(H) + a_eff(H)) - Hold
function jac(H, Hold, dt)
    Ii = Int[]; Jj = Int[]; V = Float64[]
    li = LinearIndices((N, N)); del = 1e-30
    for ci in 0:2, cj in 0:2
        E = zeros(N, N); E[ci+1:3:end, cj+1:3:end] .= 1
        dR = imag.(Res(complex.(H, del .* E), Hold, dt)) ./ del
        for j in cj+1:3:N, i in ci+1:3:N, dj in -1:1, di in -1:1
            r, s = i + di, j + dj
            (1 <= r <= N && 1 <= s <= N) || continue
            push!(Ii, li[r, s]); push!(Jj, li[i, j]); push!(V, dR[r, s])
        end
    end
    return sparse(Ii, Jj, V, N^2, N^2)
end

function be_step(Hold, dt)
    H = copy(Hold); ch = Inf
    for it in 1:cfg["max_iters"]
        R = Res(H, Hold, dt); J = jac(H, Hold, dt)
        if heps > 0
            Phi = vec(R); JJ = J
        else                                   # semismooth Newton on min(H, R)
            m = vec(H .>= R)
            Phi = ifelse.(m, vec(R), vec(H)); JJ = spdiagm(0 => Float64.(m)) * J + spdiagm(0 => Float64.(.!m))
        end
        d = reshape(lu(JJ) \ (-Phi), N, N)
        merit(Hc) = (Rc = Res(Hc, Hold, dt); heps > 0 ? norm(Rc) : norm(min.(Hc, Rc)))
        nR = norm(Phi); lam = 1.0; ok = false
        for _ in 0:8
            ok = merit(H + lam * d) <= (1 - 1e-4lam) * nR; ok && break; lam /= 2
        end
        ch = norm(d) / norm(H)
        ok || return H, ch < 1e-7, it, ch
        H = H + lam * d
        ch < cfg["tol"] && return H, true, it, ch
    end
    return H, ch < 1e-7, cfg["max_iters"], ch
end

if isempty(cfg["init"])
    H = zeros(N, N)
else                                       # bilinear interpolation of a coarser steady state
    Hc = h5read(cfg["init"], "H"); Nc = size(Hc, 1); hc = 2L / Nc
    xc = [-L + (i - 0.5) * hc for i in 1:Nc]
    itp(x) = (u = clamp((x + L) / hc + 0.5, 1, Nc); i0 = clamp(floor(Int, u), 1, Nc - 1); (i0, u - i0))
    H = [begin (i, wx) = itp(x); (j, wy) = itp(y)
            (1 - wx) * (1 - wy) * Hc[i, j] + wx * (1 - wy) * Hc[i+1, j] + (1 - wx) * wy * Hc[i, j+1] + wx * wy * Hc[i+1, j+1] end
         for x in xs, y in xs]
end
margin_r(H) = maximum(hypot(xs[i], xs[j]) for i in 1:N, j in 1:N if H[i, j] > 1.0; init = 0.0)
println(@sprintf("dense 2D EISMINT: %d^2, dx = %.2f km, heps = %g m, closure margin %.1f km", N, h / 1e3, heps, eismint_margin(eb, 2) / 1e3))
t = 0.0; dt = isempty(cfg["init"]) ? cfg["dt0"] : 1000.0; nstep = 0; nrej = 0
wall = @elapsed while t < cfg["t_max"]
    global H, t, dt, nstep, nrej
    Hn, ok, it, ch = be_step(H, dt)
    if !ok
        nrej += 1; dt /= 2; dt < 1e-3 && error("dt collapsed at t = $t"); continue
    end
    rate = norm(Hn - H) / norm(Hn) / dt
    t += dt; H = Hn; nstep += 1
    nstep % 5 == 0 && println(@sprintf("t = %8.0f a  dt %8.1f  newton %2d  H_ctr %.2f m  margin %.1f km  |dH/dt|/|H| %.1e  min H %.2e",
                                       t, dt, it, H[N÷2, N÷2], margin_r(H) / 1e3, rate, minimum(H)))
    flush(stdout)
    rate < cfg["steady_tol"] && break
    it <= 5 && (dt = min(cfg["dt_max"], 2dt))
end
println(@sprintf("steady after %d steps (%d rejected), %.0f s: H_ctr %.2f m, margin(H > 1 m) %.2f km, min H %.2e", nstep, nrej, wall,
                 H[N÷2, N÷2], margin_r(H) / 1e3, minimum(H)))
mkpath("output"); fn = "output/dense2d_n$(nb)_heps$(heps)$(cfg["tag"]).h5"
h5open(fn, "w") do f
    f["H"] = H
end
println("wrote $fn")
g = QuanticsGrid(nb, :interleaved, (-L, L), (-L, L)); D = Dx(H)
for c in parse.(Float64, split(cfg["cutoffs"], ","))
    println(@sprintf("cutoff %.0e: chi(H) %d  chi(D) %d", c, maxlinkdim(array_to_mps(H, g; cutoff = c)), maxlinkdim(array_to_mps(D, g; cutoff = c))))
end
