# Test 1: TT / QTT ranks of the fabric field psi(x, z, mu, phi) of the steady divide-to-flank flow,
# from the exact (angular central Gaussian) solution of pure lattice rotation of an isotropic
# surface fabric. Compares storage of the dense grid, a spherical-harmonic truncation of equal
# accuracy, SH + space/orientation SVD, plain TT and QTT for several bit orderings.
#
#   sbatch slurm/run.sbatch test1_ranks.jl cases="5,5,0.2" X=4
#   cases = "ns,nmu,wb;ns,nmu,wb;..."   (2^ns points per space dim, 2^nmu x 2^(nmu+1) orientations)
using Random
include("fabric_common.jl")

opts = parse_opts(Dict("cases" => "5,5,0.2", "X" => "4.0", "iota" => "-1", "eps" => "1e-2,1e-3,1e-4,1e-6",
                       "lmax" => "128", "ngl" => "192", "verify" => "1", "orderings" => "all", "sh" => "1",
                       "fs" => "0.0", "nsteps" => "400"))
X = parse(Float64, opts["X"]); iota = parse(Float64, opts["iota"]); fs = parse(Float64, opts["fs"])
epss = parse.(Float64, split(opts["eps"], ","))
lmax = parse(Int, opts["lmax"]); ngl = parse(Int, opts["ngl"]); nsteps = parse(Int, opts["nsteps"])
println("threads = $(Threads.nthreads()), X = $X, iota = $iota, fs = $fs, eps = $epss")

nsh(L) = (L + 1) * (L + 2) ÷ 2          # number of even-degree real SH coefficients up to even L

function run_case(ns, nmu, wb)
    nphi = nmu + 1
    Nx = Nz = 2^ns; Nmu = 2^nmu; Nphi = 2^nphi
    f = DivideFlow(wb = wb, fs = fs)
    xs = cellc(0.0, X, Nx); zs = cellc(0.0, 1.0, Nz)
    println("\n================ case ns=$ns nmu=$nmu (grid $(Nx)x$(Nz) x $(Nmu)x$(Nphi) = $(Nx*Nz*Nmu*Nphi) values), wb=$wb ================")
    t0 = time()
    Gs = Matrix{NTuple{4,Float64}}(undef, Nx, Nz)
    Threads.@threads for j in 1:Nz
        for i in 1:Nx
            Gs[i, j] = path_G(f, xs[i], zs[j]; iota = iota, nsteps = nsteps)
        end
    end
    Ms = map(Mfrom, Gs)
    @printf("G on space grid: %.2f s; age at bed cell %.3f\n", time() - t0, sum(1 / omega(f, z) for z in cellc(zs[1], 1.0, 2000)) * (1 - zs[1]) / 2000)
    # RK4 convergence check at the deepest, most sheared point
    Gr = path_G(f, xs[end], zs[1]; iota = iota, nsteps = 4 * nsteps)
    @printf("RK4 check (nsteps vs 4 nsteps) at (x=%.3f, z=%.4f): max |dG| = %.2e, det G = %.12f\n", xs[end], zs[1],
            maximum(abs.(Gr .- Gs[end, 1])), Gs[end, 1][1] * Gs[end, 1][4] - Gs[end, 1][2] * Gs[end, 1][3])

    # fabric strength on a GL x phi quadrature grid (also checks the normalisation)
    xq, wq = gauss_legendre(96); pq = phi_grid(192)
    a1 = zeros(Nx, Nz); a3 = zeros(Nx, Nz); nerr = zeros(Nx, Nz)
    Threads.@threads for j in 1:Nz
        for i in 1:Nx
            F = [psi_closed(Ms[i, j], m, p) for m in xq, p in pq]
            ev, tot = a2_eigs(F, xq, wq, pq)
            a1[i, j] = ev[1]; a3[i, j] = ev[3]; nerr[i, j] = abs(tot - 1)
        end
    end
    imax = argmax(a1)
    @printf("fabric: a1 in [%.3f, %.3f], min a3 = %.3e, max a1 at (x=%.2f, z=%.3f); max |int psi - 1| = %.2e (GL 96x192)\n",
            minimum(a1), maximum(a1), minimum(a3), xs[imax[1]], zs[imax[2]], maximum(nerr))
    q = sort(vec(a1)); @printf("a1 quantiles: 10%% %.3f, 50%% %.3f, 90%% %.3f\n", q[ceil(Int, 0.1end)], q[ceil(Int, 0.5end)], q[ceil(Int, 0.9end)])

    if opts["verify"] == "1"
        # Monte Carlo push-forward of an isotropic sample under n = G n0 / |G n0| at the strongest point
        G = Gs[imax]; Ns = 4_000_000; nb_mu, nb_phi = 16, 32
        cnt = zeros(nb_mu, nb_phi); rng = MersenneTwister(1)
        for _ in 1:Ns
            v = randn(rng, 3)                                # isotropic direction (x, y, z)
            nx = G[1] * v[1] + G[2] * v[3]; nz = G[3] * v[1] + G[4] * v[3]; ny = v[2]
            r = sqrt(nx^2 + ny^2 + nz^2); mu = ny / r; ph = mod(atan(nz, nx), 2pi)
            cnt[min(nb_mu, 1 + floor(Int, (mu + 1) / 2 * nb_mu)), min(nb_phi, 1 + floor(Int, ph / 2pi * nb_phi))] += 1
        end
        hm, hp = 2 / nb_mu, 2pi / nb_phi
        est = cnt ./ (Ns * hm * hp)
        sub = 32
        MG = Mfrom(G)
        ex = [sum(psi_closed(MG, -1 + (i - 1 + (a - 0.5) / sub) * hm, (j - 1 + (b - 0.5) / sub) * hp) for a in 1:sub, b in 1:sub) / sub^2
              for i in 1:nb_mu, j in 1:nb_phi]
        staterr = sqrt.(max.(cnt, 1)) ./ (Ns * hm * hp)
        @printf("Monte Carlo check (4e6 samples, %dx%d bins): rel L2 diff %.3e, max |diff|/stat.err %.2f, rel stat err (L2) %.3e\n",
                nb_mu, nb_phi, norm(est - ex) / norm(ex), maximum(abs.(est - ex) ./ staterr), norm(staterr) / norm(ex))
    end

    mus = mu_grid(Nmu); phis = phi_grid(Nphi)
    t0 = time()
    A = dense_psi(Ms, mus, phis)                          # (Nx, Nz, Nmu, Nphi)
    @printf("dense psi: %.2f s, %.1f MB, max psi %.3e, min psi %.3e\n", time() - t0, sizeof(A) / 2^20, maximum(A), minimum(A))
    Ntot = length(A)
    lm = map(Meigs, Ms)
    lmin = minimum(first.(lm)); il = argmin(first.(lm)); lmx = lm[il][2]
    gsum = dropdims(sum(A, dims = (3, 4)), dims = (3, 4)) .* (2 / Nmu) .* (2pi / Nphi)
    @printf("sharpest fabric: lam_min %.3e, lam_max %.3e -> peak psi %.2f, half-width ~ %.2f deg (grid h_phi = %.2f deg, h_theta at equator = %.2f deg); grid normalisation max |sum psi dA - 1| = %.2e\n",
            lmin, lmx, lmin^-1.5 / 4pi, rad2deg(sqrt(lmin / lmx)), 360 / Nphi, rad2deg(2 / Nmu), maximum(abs.(gsum .- 1)))

    # space | orientation separation spectrum
    t0 = time()
    sv = svdvals(reshape(A, Nx * Nz, Nmu * Nphi))
    tail = sqrt.(reverse(cumsum(reverse(sv .^ 2)))) ./ norm(sv)     # tail[i] = rel. error keeping i-1 terms
    rsep(e) = findfirst(i -> (i == length(sv) ? 0.0 : tail[i+1]) <= e, 1:length(sv))
    @printf("space|orientation SVD (%.1f s): sigma_k/sigma_1 at k=1,2,4,8,16,32,64,128: %s\n", time() - t0,
            join([@sprintf("%.1e", sv[k] / sv[1]) for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= length(sv)], " "))
    println("RESULT_SEPSPEC ns=$ns nmu=$nmu wb=$wb ", join([@sprintf("%.4e", s / sv[1]) for s in sv[1:min(256, end)]], ","))

    # spherical harmonics on a subsample of the space grid
    shL = Dict{Float64,Int}()
    if opts["sh"] == "1"
        t0 = time()
        S, xq2, pq2 = make_sh(lmax, ngl, 2ngl)
        stride = max(1, Nx ÷ 32)
        pts = [(i, j) for i in 1:stride:Nx for j in 1:stride:Nz]
        Es = sh_energies(S, p -> [psi_closed(Ms[pts[p]...], m, ph) for m in xq2, ph in pq2], length(pts), ngl, 2ngl)
        Etot = vec(sum(Es, dims = 2))
        @printf("SH (lmax=%d, GL %d x %d, %d space points, %.1f s): odd-degree energy fraction %.1e, energy fraction beyond l=%d: %.1e (worst point %.1e), Parseval vs GL norm: %.1e\n",
                lmax, ngl, 2ngl, length(pts), time() - t0, sum(Etot[2:2:end]) / sum(Etot), lmax - 8, sum(Etot[lmax-7:end]) / sum(Etot),
                maximum(sum(Es[lmax-7:end, :], dims = 1) ./ sum(Es, dims = 1)),
                abs(sum(Etot) / sum(sum(S.w[k] * psi_closed(Ms[pts[p]...], xq2[k], ph)^2 * S.hphi for k in eachindex(xq2), ph in pq2) for p in eachindex(pts)) - 1))
        rel = sqrt.(max.(0.0, reverse(cumsum(reverse(Etot)))) ./ sum(Etot))     # rel[l+1] = error of truncation at degree l-1
        for e in epss
            L = findfirst(l -> iseven(l) && (l + 2 > lmax + 1 ? 0.0 : rel[l+2]) <= e, 0:lmax)
            shL[e] = L === nothing ? -1 : L - 1
        end
        println("RESULT_SHSPEC ns=$ns nmu=$nmu wb=$wb ", join([@sprintf("%.4e", v) for v in rel], ","))
    end

    # orderings (space dims 1:2 = x, z; orientation dims 3:4 = mu, phi)
    nb = [ns, ns, nmu, nphi]
    orders = Dict(
        "seq(x,z,mu,phi)"      => seq_order([1, 2, 3, 4], nb),
        "seq(mu,phi,x,z)"      => seq_order([3, 4, 1, 2], nb),
        "inter(x,z,mu,phi)"    => inter_order([1, 2, 3, 4], nb),
        "inter(x,z)|inter(mu,phi)" => [inter_order([1, 2], nb); inter_order([3, 4], nb)],
        "inter(mu,phi)|inter(x,z)" => [inter_order([3, 4], nb); inter_order([1, 2], nb)],
        "seq(x,z)|inter(mu,phi)"   => [seq_order([1, 2], nb); inter_order([3, 4], nb)],
    )
    onames = ["seq(x,z,mu,phi)", "seq(mu,phi,x,z)", "inter(x,z,mu,phi)", "inter(x,z)|inter(mu,phi)", "inter(mu,phi)|inter(x,z)", "seq(x,z)|inter(mu,phi)"]
    opts["orderings"] != "all" && (onames = onames[parse.(Int, split(opts["orderings"], ","))])

    println("\nstorage (number of Float64), dense = $Ntot")
    @printf("%-28s %8s %10s %8s %12s %10s\n", "representation", "eps", "maxrank", "midrank", "params", "compress")
    for e in epss
        r = rsep(e)
        @printf("%-28s %8.0e %10d %8s %12d %10.1f\n", "space|orient SVD (grid)", e, r, "-", r * (Nx * Nz + Nmu * Nphi), Ntot / (r * (Nx * Nz + Nmu * Nphi)))
        if haskey(shL, e) && shL[e] >= 0
            L = shL[e]; nS = nsh(L)
            @printf("%-28s %8.0e %10s %8s %12d %10.1f   (L = %d, %d coeffs/point)\n", "SH dense space", e, "-", "-", Nx * Nz * nS, Ntot / (Nx * Nz * nS), L, nS)
            # SH + SVD over space: rank of the coefficient matrix equals the separation rank (basis change)
            @printf("%-28s %8.0e %10d %8s %12d %10.1f\n", "SH + space SVD", e, r, "-", r * (Nx * Nz + nS), Ntot / (r * (Nx * Nz + nS)))
        end
        t0 = time()
        rk, np, _ = tt_svd(A, e)
        @printf("%-28s %8.0e %10d %8s %12d %10.1f   ranks %s (%.1f s)\n", "TT (x,z,mu,phi)", e, maximum(rk), "-", np, Ntot / np, rk, time() - t0)
        for name in onames
            t0 = time()
            T = quantics_tensor(A, nb, orders[name])
            rk, np, _ = tt_svd(T, e)
            nsp = 2ns
            mid = name in ("seq(x,z,mu,phi)", "inter(x,z)|inter(mu,phi)", "seq(x,z)|inter(mu,phi)") ? string(rk[nsp]) :
                  name in ("seq(mu,phi,x,z)", "inter(mu,phi)|inter(x,z)") ? string(rk[nmu+nphi]) : "-"
            @printf("%-28s %8.0e %10d %8s %12d %10.1f   (%.1f s)\n", "QTT " * name, e, maximum(rk), mid, np, Ntot / np, time() - t0)
            println("RESULT_RANKS ns=$ns nmu=$nmu wb=$wb eps=$e order=$name ranks=", join(rk, ","))
        end
    end

    # orientation-only QTT of the strongest single fabric (per-point comparison with SH)
    i, j = Tuple(imax)
    Fo = A[i, j, :, :]
    for e in epss
        rk1, np1, _ = tt_svd(quantics_tensor(Fo, [nmu, nphi], inter_order([1, 2], [nmu, nphi])), e)
        rk2, np2, _ = tt_svd(quantics_tensor(Fo, [nmu, nphi], seq_order([1, 2], [nmu, nphi])), e)
        @printf("single strongest fabric (a1=%.3f), eps %.0e: QTT inter max rank %d (%d params), seq max rank %d (%d params), grid %d, SH L=%s (%s coeffs)\n",
                a1[imax], e, maximum(rk1), np1, maximum(rk2), np2, Nmu * Nphi, get(shL, e, -1), haskey(shL, e) && shL[e] >= 0 ? nsh(shL[e]) : "-")
    end
    return nothing
end

for c in split(opts["cases"], ";")
    a = split(c, ",")
    run_case(parse(Int, a[1]), parse(Int, a[2]), parse(Float64, a[3]))
end
println("done")
