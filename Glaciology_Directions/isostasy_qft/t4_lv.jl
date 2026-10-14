# Task 4: laterally variable viscosity (LV-ELVA, FastIsostasy's lv_elva! with constant D):
#     du/dt = -|k|^{-1} [ (1/(2 eta(x))) (A u + sigma) ],   A = rho_m g + D lap^2  (Fourier diagonal)
# Not diagonal in either space. Scheme: exponential time differencing (ETD1) about a reference
# viscosity eta0 <= min eta (r = eta0/eta in (0,1]):
#     u^{n+1} = u^n - (1 - E0) A^{-1} [ r .* (A u^n + sigma) ],   E0 = exp(-lambda0 dt)
# Exact for r = 1; steady state A u = -sigma for any r; amplification I - (1-E0) r has eigenvalues
# in (0, 1] for every dt and every resolution (no CFL). u is kept in Fourier space: per step one
# inverse QFT (for z = A u + sigma), one Hadamard product with r, one QFT, two diagonal multipliers.
# Constant mode: FastIsostasy's corner condition (mean of the 4 corner displacements = 0).
#
# Reference: explicit RK4 (FastIsostasy integrates lv_elva! with explicit Tsit5) at small dt.
# usage: julia t4_lv.jl nref=7 nlist=7,8 dts=10,50,100 dtmps=50 tend=5000 eps=1e-7 ordering=interleaved
include("isostasy.jl")
using Statistics
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1"))); FFTW.set_num_threads(1)

opts = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
parseints(s) = isempty(s) ? Int[] : parse.(Int, split(s, ","))
nref = parse(Int, get(opts, "nref", "7"))
nlist = parseints(get(opts, "nlist", "7,8"))
dts = parse.(Float64, split(get(opts, "dts", "1,5,10,50,100,500"), ","))
dtmps = parse(Float64, get(opts, "dtmps", "50"))
tend = parse(Float64, get(opts, "tend", "5000"))
epsv = parse(Float64, get(opts, "eps", "1e-7"))
ord = Symbol(get(opts, "ordering", "interleaved"))
nmpsdense = parse(Int, get(opts, "nmpsdense", "12"))   # dense ETD comparison up to this n
cut = epsv^2
p = Earth()
W = 3000e3; L = 2W; R0 = 1000e3; H0 = 1000.0; wd = 20e3
eta0 = 1e19
etaf(x, y) = 10.0^(21 - 2 * exp(-((x - 800e3)^2 + y^2) / (400e3)^2))   # 1e19 anomaly under the disc edge
rf(x, y) = eta0 / etaf(x, y)
loadf(x, y) = p.rho_ice * p.g * disc_smooth(x, y; R0 = R0, H0 = H0, w = wd)
touts = [500.0, 1000.0, 2000.0, 5000.0][[500.0, 1000.0, 2000.0, 5000.0] .<= tend]
@printf("LV-ELVA: eta in [1e19, 1e21], eta0 = %.0e, smooth disc R0 = 1000 km, outputs %s yr\n", eta0, touts)
flush(stdout)

# ---------------- dense helpers ----------------
struct Dense
    N::Int; K::Matrix{Float64}; B::Matrix{Float64}; Kinv::Matrix{Float64}
    r::Matrix{Float64}; sig::Matrix{Float64}; P; IP
end
function Dense(n)
    N = 2^n; x = gridpoints(N, W)
    K = kgrid2(N, L); B = beta.(K, Ref(p)); Kinv = [k == 0 ? 0.0 : 1 / k for k in K]
    r = [rf(a, b) for a in x, b in x]; sig = [loadf(a, b) for a in x, b in x]
    Z = zeros(ComplexF64, N, N)
    Dense(N, K, B, Kinv, r, sig, plan_fft(Z; flags = FFTW.ESTIMATE), plan_ifft(Z; flags = FFTW.ESTIMATE))
end
# corner value from Fourier coefficients (ifft convention), O(N^2)
function corner_from_hat(D::Dense, uh)
    N = D.N
    ph = [1.0 + 0im, 0.0im]; v0 = ones(ComplexF64, N); v1 = [cis(2pi * m * (N - 1) / N) for m in 0:N-1]
    c = 0.0
    for a in (v0, v1), b in (v0, v1)
        c += real(transpose(a) * uh * b)
    end
    return c / 4 / N^2
end
# RK4 on du/dt = -K^{-1}[(r/(2 eta0)) (A u + sigma)], constant mode slaved to the corner condition
function rhs!(D::Dense, u)
    corner_bc!(u)
    z = real.(D.IP * (D.B .* (D.P * complex.(u)))) .+ D.sig
    w = (D.r ./ (2eta0)) .* z
    return -real.(D.IP * (D.Kinv .* (D.P * complex.(w)))) .* SEC_PER_YEAR    # per year
end
function rk4(D::Dense, touts, dt)
    u = zeros(D.N, D.N); t = 0.0; out = Dict{Float64,Matrix{Float64}}()
    for to in touts
        while t < to - 1e-9
            h = min(dt, to - t)
            k1 = rhs!(D, u); k2 = rhs!(D, u .+ h / 2 .* k1); k3 = rhs!(D, u .+ h / 2 .* k2); k4 = rhs!(D, u .+ h .* k3)
            u .+= h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4); t += h
        end
        out[to] = corner_bc!(copy(u))
    end
    return out
end
function etd_dense(D::Dense, touts, dt)
    lam0 = [k == 0 ? Inf : beta(k, p) / (2eta0 * k) for k in D.K]
    E0 = exp.(-lam0 .* dt .* SEC_PER_YEAR)
    G0 = [k == 0 ? 0.0 : g for (k, g) in zip(D.K, (1 .- E0) ./ D.B)]
    uh = zeros(ComplexF64, D.N, D.N); t = 0.0; out = Dict{Float64,Matrix{Float64}}()
    nsteps = 0; tstep = 0.0
    for to in touts
        while t < to - 1e-9
            t0 = time()
            z = real.(D.IP * (D.B .* uh)) .+ D.sig
            wh = D.P * complex.(D.r .* z)
            uh .-= G0 .* wh
            uh[1, 1] -= D.N^2 * corner_from_hat(D, uh)
            tstep += time() - t0; nsteps += 1
            t += dt
        end
        out[to] = real.(D.IP * uh)
    end
    return out, tstep / nsteps
end
rel(a, b) = norm(a - b) / norm(b)

# ---------------- 1. dense: ETD1 vs explicit RK4 reference ----------------
println("\n== 1. dense n = $nref: ETD1(dt) vs explicit RK4 reference ==")
flush(stdout)
D = Dense(nref)
lam_max = maximum(D.B[2:end] ./ (2eta0 .* D.K[2:end])) * SEC_PER_YEAR    # 1/yr, fastest rate
dtref = 1.0 / lam_max
(ref, tr) = timed(() -> rk4(D, touts, dtref))
@printf("RK4: dt = %.3e yr (1/lambda_max at eta_min), %d steps, %.1f s; max|u| at t_end %.1f m\n", dtref, ceil(Int, tend / dtref), tr, maximum(abs.(ref[touts[end]])))
flush(stdout)
# constant-viscosity sanity check: with r = 1 ETD1 must equal the exact exponential solution
for dt in dts
    (sol, ts) = etd_dense(D, touts, dt)
    @printf("ETD1 dt=%6.1f yr: rel L2 err vs RK4 at t=%s : %s   (%.2e s/step)\n", dt, touts,
            join([@sprintf("%.2e", rel(sol[t], ref[t])) for t in touts], " "), ts)
    flush(stdout)
end

# ---------------- 2. MPS ETD1 vs dense ETD1 ----------------
println("\n== 2. MPS ETD1 (dt = $dtmps yr, eps = $epsv, ordering $ord) vs dense ETD1 ==")
flush(stdout)
for n in nlist
    N = 2^n
    g = QGrid(2, n; ordering = ord)
    tb = time()
    Pp = Pipeline(g; qcut = cut * 1e-4)
    lam0(k) = k == 0 ? Inf : beta(k, p) / (2eta0 * k)
    mB = mult_mps(k -> beta(k, p), g, L; scale = 1 / N^2, tolerance = epsv * 1e-3, cutoff = cut * 1e-4)
    mG = mult_mps(k -> k == 0 ? 0.0 : (1 - exp(-lam0(k) * dtmps * SEC_PER_YEAR)) / beta(k, p), g, L; tolerance = epsv * 1e-3, cutoff = cut * 1e-4)
    rr, _, _ = tci_mps(idx -> rf(-W + idx[1] * L / N, -W + idx[2] * L / N), g; T = ComplexF64, tolerance = epsv * 1e-2)
    truncate!(rr; cutoff = cut * 1e-2)
    ss, _, _ = tci_mps(idx -> loadf(-W + idx[1] * L / N, -W + idx[2] * L / N), g; T = ComplexF64, tolerance = epsv * 1e-2)
    truncate!(ss; cutoff = cut * 1e-2)
    corners = [phase_state(g, (a, b)) for a in (0, N - 1), b in (0, N - 1)]
    e0 = zero_mode_state(g)
    tsetup = time() - tb
    @printf("n=%2d: setup %.1fs | chi: Q %d, beta %d, G0 %d, r %d, sigma %d\n", n, tsetup, maxlinkdim(Pp.Q), maxlinkdim(mB), maxlinkdim(mG), maxlinkdim(rr), maxlinkdim(ss))
    flush(stdout)
    uh = nothing; t = 0.0; tsteps = Float64[]; chis = Int[]
    Dn = n <= nmpsdense ? Dense(n) : nothing
    dsol = Dn === nothing ? nothing : etd_dense(Dn, touts, dtmps)
    for to in touts
        while t < to - 1e-9
            t0 = time()
            if uh === nothing
                z = ss
            else
                zz = apply(Pp.Qa, apply(diag_mpo(mB), uh; alg = "naive", cutoff = cut); alg = "naive", cutoff = cut)
                z = lincomb([0.5, 0.5, 1.0], [zz, dag(zz), ss]; cutoff = cut)
            end
            w = apply(diag_mpo(rr), z; alg = "naive", cutoff = cut)
            wh = apply(Pp.Q, w; alg = "naive", cutoff = cut)
            d = apply(diag_mpo(mG), wh; alg = "naive", cutoff = cut)
            uh = uh === nothing ? -1.0 * d : lincomb([1.0, -1.0], [uh, d]; cutoff = cut)
            c = sum(real(inner(cs, uh)) for cs in corners) / 4      # = N^2 * corner mean
            uh = lincomb([1.0, -c], [uh, e0]; cutoff = cut)
            push!(tsteps, time() - t0); push!(chis, maxlinkdim(uh))
            t += dtmps
        end
        u = apply(Pp.Qa, uh; alg = "naive", cutoff = cut) * (1 / N^2)
        u = realpart(u; cutoff = cut)
        if dsol !== nothing
            U = dsol[1][to]
            Um = real.(mps_to_array(u, g))
            @printf("   t=%5.0f: chi(uh) %3d chi(u) %3d | MPS %.2fs/step (dense %.3fs/step) | rel err vs dense ETD1 %.2e (max %.2e m)\n",
                    to, maxlinkdim(uh), maxlinkdim(u), median(tsteps), dsol[2], rel(Um, U), maximum(abs.(Um - U)))
            flush(stdout)
        else
            @printf("   t=%5.0f: chi(uh) %3d chi(u) %3d | MPS %.2fs/step | u(0,0) = %.3f m\n", to, maxlinkdim(uh), maxlinkdim(u), median(tsteps),
                    real(MPSEval(u)(idx_to_bits(g, (N ÷ 2, N ÷ 2)))))
            flush(stdout)
        end
    end
end
println("done")
flush(stdout)
