# Do ITensors' default add / apply become exact with a tight cutoff? Residual of a Halfar step
#   R = H1 - dt F(H1) - H0
# formed with add(...; cutoff) (default algorithm "densitymatrix") at several cutoffs vs the exact
# direct sum; and F with apply(...; cutoff) (default "densitymatrix") vs naive contraction.
# Compared with the sparse-matrix value on the same arrays.
#   sbatch --qos=30min --time=00:29:00 slurm/run.sbatch bench_sum.jl bits=12,14,16
using LinearAlgebra, SparseArrays, Printf
include("quantics.jl")
include("sia.jl")
include("benchmarks.jl")

opts = Dict("bits" => "12,14,16", "dt" => "2.5")
for a in ARGS
    k, v = split(a, "="); opts[k] = v
end
bits = parse.(Int, split(opts["bits"], ","))
dt = parse(Float64, opts["dt"])
p = SIAParams(); hb = Halfar(d = 1, p = p); G = Gamma(p); t0 = halfar_t0(hb)

for nbits in bits
    g = QuanticsGrid(nbits, :interleaved, (-1200e3, 1200e3)); ops = SIAOperators(g)
    H0 = function_to_mps(x -> halfar_H(hb, t0, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    H1 = function_to_mps(x -> halfar_H(hb, t0 + dt, abs(x)), g; tolerance = 1e-8, maxbonddim = 128)
    N = 2^nbits; h = g.h[1]
    Sp = spdiagm(1 => ones(N - 1)); Df = (Sp - I) / h; Db = (I - Sp') / h; Avg = (I + Sp) / 2
    h0 = mps_to_array(H0, g); h1 = mps_to_array(H1, g)
    He = Avg * h1; gg = Df * h1
    Fs = Db * (G .* He .^ 5 .* gg .^ 3)
    Rs = h1 - dt * Fs - h0
    println(@sprintf("\n2^%d points: |R|/|H1| = %.1e", nbits, norm(Rs) / norm(h1)))
    # F with each product algorithm (sums exact)
    for (alg, cut) in (("densitymatrix", 1e-15), ("densitymatrix", 1e-30), ("naive", 1e-24))
        PRODUCT_ALG[] = alg
        st = SolverSettings(prod_cutoff = cut)
        F = flux_divergence(face_fields(H1, nothing, ops, p, st), ops, st)
        Fm = mps_to_array(F, g)
        println(@sprintf("   F via apply alg=%-13s cutoff %.0e: rel err %.1e", alg, cut, norm(Fm - Fs) / norm(Fs)))
        flush(stdout)
    end
    PRODUCT_ALG[] = "naive"
    st = SolverSettings()
    F = flux_divergence(face_fields(H1, nothing, ops, p, st), ops, st)
    terms = [H1, -dt * F, -1.0 * H0]
    errR(R) = norm(mps_to_array(R, g) - Rs) / norm(Rs)
    for cut in (nothing, 1e-15, 1e-20, 1e-30)
        R = cut === nothing ? add(add(terms[1], terms[2]), terms[3]) :
                              add(add(terms[1], terms[2]; cutoff = cut), terms[3]; cutoff = cut)
        println(@sprintf("   R via add (default alg)  cutoff %-7s: rel err %.1e  chi %d", cut === nothing ? "default" : @sprintf("%.0e", cut), errR(R), maxlinkdim(R)))
    end
    R = lincomb([1.0, 1.0, 1.0], terms; cutoff = 1e-30)
    println(@sprintf("   R via exact direct sum + SVD      : rel err %.1e  chi %d", errR(R), maxlinkdim(R)))
    flush(stdout)
end
