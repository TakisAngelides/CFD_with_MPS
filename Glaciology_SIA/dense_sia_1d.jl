# Dense (sparse-matrix) reference implementation of exactly the same 1D discrete scheme as the MPS
# solver (Mahaffy stencil, Picard, BE/BDF2), to separate discretisation behaviour from MPS errors.
#   sbatch slurm/run.sbatch dense_sia_1d.jl nbits=12 scheme=bdf2 dt=100 t_run=10000

using LinearAlgebra, SparseArrays, Printf, HDF5
include("quantics.jl")      # only for SIAParams / Gamma via sia.jl
include("sia.jl")
include("benchmarks.jl")

cfg = Dict{String,Any}("nbits" => 12, "scheme" => "bdf2", "dt" => 100.0, "t_run" => 10000.0, "max_iters" => 30, "nonlinear" => "newton",
                       "tol" => 1e-10, "experiment" => "halfar", "outdir" => joinpath(@__DIR__, "output"))
for arg in ARGS
    k, v = split(arg, "=", limit = 2)
    cfg[k] = cfg[k] isa String ? String(v) : parse(typeof(cfg[k]), v)
end

p = SIAParams()
N = 2^cfg["nbits"]
if cfg["experiment"] == "halfar"
    hb = Halfar(d = 1, p = p); L = 1200e3; t_start = halfar_t0(hb); a = zeros(N)
else
    eb = EISMINTMovingMargin(p = p); L = eb.L; t_start = 0.0
end
h = 2L / N
x = [-L + (i + 0.5) * h for i in 0:N-1]
cfg["experiment"] == "eismint" && (a = [eismint_mb(eb, abs(xi)) for xi in x])

Sp = spdiagm(1 => ones(N - 1))           # (Sp u)_i = u_{i+1}, zero ghost
Df = (Sp - I) / h
Db = (I - Sp') / h
Avg = (I + Sp) / 2

G = Gamma(p)
@assert p.n_glen == 3 "dense Newton Jacobian written for n = 3"
F(H) = Db * (Dface(H) .* (Df * H))
Jac(H) = (He = Avg * H; g = Df * H;
          Db * (spdiagm(0 => 3G .* He .^ 5 .* g .^ 2) * Df + spdiagm(0 => 5G .* He .^ 4 .* g .^ 3) * Avg))

function Dface(H)
    He = Avg * H
    s = Df * H
    return Gamma(p) .* He .^ (p.n_glen + 2) .* (s .^ 2) .^ ((p.n_glen - 1) ÷ 2)
end

H = cfg["experiment"] == "halfar" ? [halfar_H(hb, t_start, abs(xi)) for xi in x] : max.(a, 0.0) .* cfg["dt"]
H_prev = nothing
t = t_start
dt = cfg["dt"]
times = [t]; Hc = [H[N÷2+1]]; errs = Float64[NaN]; wall = Float64[]
for step in 1:round(Int, cfg["t_run"] / dt)
    global H, H_prev, t
    t += dt
    wt = @elapsed begin
        if cfg["scheme"] == "bdf2" && H_prev !== nothing
            rhs0 = (4H - H_prev) / 3 + (2dt / 3) * a; dte = 2dt / 3
        else
            rhs0 = H + dt * a; dte = dt
        end
        Hn = copy(H); change = Inf; it = 0
        Res(H) = H - dte * F(H) - rhs0
        for m in 1:cfg["max_iters"]
            it = m
            if cfg["nonlinear"] == "picard" || m == 1
                Hnew = (I - dte * Db * spdiagm(0 => Dface(Hn)) * Df) \ rhs0
            else
                # damped Newton (same as the MPS solver), H >= 0 enforced in the line search
                R = Res(Hn); nR = norm(R)
                d = try lu(I - dte * Jac(Hn)) \ R catch; Hn - (I - dte * Db * spdiagm(0 => Dface(Hn)) * Df) \ rhs0 end
                lam = 1.0; Hnew = max.(Hn - d, 0.0)
                while norm(Res(Hnew)) > (1 - 1e-4lam) * nR && lam > 1e-4
                    lam /= 2; Hnew = max.(Hn - lam * d, 0.0)
                end
            end
            change = norm(Hnew - Hn) / norm(Hn)
            Hn = Hnew
            change < cfg["tol"] && break
        end
        cfg["experiment"] == "eismint" && (Hn = max.(Hn, 0.0))
        H_prev, H = H, Hn
    end
    push!(wall, wt); push!(times, t); push!(Hc, H[N÷2+1])
    err = cfg["experiment"] == "halfar" ? maximum(abs, H - [halfar_H(hb, t, abs(xi)) for xi in x]) : NaN
    push!(errs, err)
    ectr = cfg["experiment"] == "halfar" ? abs(H[N÷2+1] - halfar_H(hb, t, abs(x[N÷2+1]))) : NaN
    println(@sprintf("step %4d t = %8.0f a  H_ctr %7.1f m  min H %.2e  %.3f s  err_max %.2f m  err_ctr %.2f m",
                     step, t, H[N÷2+1], minimum(H), wt, err, ectr))
end
mkpath(cfg["outdir"])
h5open(joinpath(cfg["outdir"], @sprintf("dense1d_%s_n%d_%s_dt%g.h5", cfg["experiment"], cfg["nbits"], cfg["scheme"], dt)), "w") do f
    f["times"] = times; f["H_center"] = Hc; f["err_max"] = errs; f["walltime"] = wall; f["x"] = x; f["H_final"] = H
end
