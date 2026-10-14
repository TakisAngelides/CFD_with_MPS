# Test 1b: one fabric (orientation only) of increasing strength. QTT of psi(mu, phi) on refined
# grids vs the spherical-harmonic truncation of equal L2 accuracy and the grid itself.
#   sbatch slurm/run.sbatch test1b_single.jl nmax=10
include("fabric_common.jl")

opts = parse_opts(Dict("lfac" => "12", "nmin" => "6", "nmax" => "10", "eps" => "1e-2,1e-3,1e-4,1e-6",
                       "cases" => "pure:0.5,pure:1.0,pure:1.5,pure:2.0,simple:1,simple:2,simple:4,simple:8,uni:0.5,uni:1.0,uni:1.5,uni:2.0"))
epss = parse.(Float64, split(opts["eps"], ","))
nmin, nmax = parse(Int, opts["nmin"]), parse(Int, opts["nmax"])
nsh(L) = (L + 1) * (L + 2) ÷ 2

# M3 = B^T B (B = G^{-1}) as (xx, xy, xz, yy, yz, zz), c-axes (iota = -1)
function fabric_M3(kind, s)
    if kind == "pure"            # plane strain: x extension, z compression; G = diag(e^-s, 1, e^s)
        return (exp(2s), 0.0, 0.0, 1.0, 0.0, exp(-2s))
    elseif kind == "uni"         # uniaxial z compression; G = diag(e^{-s/2}, e^{-s/2}, e^s)
        return (exp(s), 0.0, 0.0, exp(s), 0.0, exp(-2s))
    elseif kind == "simple"      # simple shear du/dz = 1 for time s: K = -L^T, G = [[1, 0], [-s, 1]] (x, z)
        G = (1.0, 0.0, -s, 1.0); M = Mfrom(G)
        return (M[1], 0.0, M[2], 1.0, 0.0, M[3])
    elseif kind == "watson"      # regularised-like reference: psi ~ exp(kappa n_z^2) (kappa < 0: girdle)
        return nothing
    end
    error("unknown fabric $kind")
end

println("threads $(Threads.nthreads())")
for c in split(opts["cases"], ",")
    kind, sv = split(c, ":"); s = parse(Float64, sv)
    M = fabric_M3(kind, s)
    if M === nothing
        kap = s
        f = (mu, phi) -> exp(kap * ((1 - mu^2) * sin(phi)^2 - 1)) / (4pi)
        ev = [1.0, 1.0, max(abs(kap), 1.0)]            # width ~ 1/sqrt(kappa) for the lmax choice
    else
        ev = eigvals(Symmetric([M[1] M[2] M[3]; M[2] M[4] M[5]; M[3] M[5] M[6]]))
        f = (mu, phi) -> psi_closed3(M, mu, phi)
    end
    w = sqrt(ev[1] / ev[3])
    # fabric eigenvalues and SH spectrum on a GL grid that resolves the peak
    lmax = min(1600, round(Int, parse(Float64, opts["lfac"]) / w) + 32); lmax += isodd(lmax)
    Ngl = round(Int, 1.5lmax) + 32; Nph = 2 * round(Int, 1.5lmax) + 64
    x, wq = gauss_legendre(Ngl); ph = phi_grid(Nph)
    F = [f(m, p) for m in x, p in ph]
    a, tot = a2_eigs(F, x, wq, ph); a = a ./ tot
    t0 = time()
    E = sh_energy_stream(F, x, wq, 2pi / Nph, lmax)
    rel = sqrt.(max.(0.0, reverse(cumsum(reverse(E)))) ./ sum(E))
    pars = abs(sum(E) / sum(wq[k] * F[k, j]^2 * 2pi / Nph for k in 1:Ngl, j in 1:Nph) - 1)
    Ls = Dict(e => (L = findfirst(l -> iseven(l) && (l + 2 > lmax + 1 ? 0.0 : rel[l+2]) <= e, 0:lmax); L === nothing ? -1 : L - 1) for e in epss)
    @printf("\n=== %s s=%.2f: a = (%.3f, %.3f, %.3f), lam_min/lam_max = %.2e, half-width %.2f deg, peak psi %.1f; int psi - 1 = %.1e\n",
            kind, s, a..., ev[1] / ev[3], rad2deg(w), ev[1]^-1.5 / 4pi, tot - 1)
    @printf("    SH: lmax %d (GL %d x %d, %.1f s), odd energy %.1e, tail beyond lmax-8 %.1e, Parseval %.1e; L(eps) = %s\n",
            lmax, Ngl, Nph, time() - t0, sum(E[2:2:end]) / sum(E), sum(E[end-7:end]) / sum(E), pars,
            join(["$(e)=>$(Ls[e]) ($(Ls[e] >= 0 ? nsh(Ls[e]) : -1))" for e in epss], ", "))
    println("RESULT_SH1 $kind $s ", join([@sprintf("%.4e", v) for v in rel], ","))
    for n in nmin:nmax
        Nmu, Nphi = 2^n, 2^(n + 1)
        mus, phis = mu_grid(Nmu), phi_grid(Nphi)
        A = [f(m, p) for m in mus, p in phis]
        gn = sum(A) * (2 / Nmu) * (2pi / Nphi) - 1
        sv = svdvals(A); tail = sqrt.(reverse(cumsum(reverse(sv .^ 2)))) ./ norm(sv)
        line = @sprintf("    n=%2d grid %5dx%5d (%9d), grid norm err %8.1e |", n, Nmu, Nphi, Nmu * Nphi, gn)
        for e in epss
            r2 = findfirst(i -> (i == length(sv) ? 0.0 : tail[i+1]) <= e, 1:length(sv))
            rkI, npI, _ = tt_svd(quantics_tensor(A, [n, n + 1], inter_order([1, 2], [n, n + 1])), e)
            rkS, npS, _ = tt_svd(quantics_tensor(A, [n, n + 1], seq_order([1, 2], [n, n + 1])), e)
            line *= @sprintf(" eps %.0e: QTTi r%3d p%6d, QTTs r%3d p%6d, svd r%3d |", e, maximum(rkI), npI, maximum(rkS), npS, r2)
            println("RESULT_1B $kind $s n=$n eps=$e qtti_rank=$(maximum(rkI)) qtti_par=$npI qtts_rank=$(maximum(rkS)) qtts_par=$npS svd_rank=$r2 sh_L=$(Ls[e]) sh_par=$(Ls[e] >= 0 ? nsh(Ls[e]) : -1) gridnorm=$gn")
        end
        println(line)
    end
end
println("done")
