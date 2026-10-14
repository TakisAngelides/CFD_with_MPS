# Test 2: explicit RK4 time stepping of the lattice-rotation (+ sphere diffusion) equation as
# MPO x MPS products, versus a dense stepper with the identical discretisation and the closed form.
# Local deformation (no spatial advection): every space point r deforms with the velocity gradient of
# the divide-to-flank flow, K(r) constant in time, from an isotropic fabric at t = 0.
#   space = "z"  : 1 space dim (column at x = x0), 3D problem (z, mu, phi)
#   space = "xz" : 2 space dims, 4D problem (x, z, mu, phi), x in [0, X]
#   sbatch slurm/run.sbatch stepper.jl space=z nz=6 nmu=5 T=0.3
include("fabric_mps.jl")

opts = parse_opts(Dict("space" => "z", "nz" => "6", "nx" => "6", "nmu" => "5", "x0" => "1.5", "X" => "3.0", "wb" => "0.3",
                       "iota" => "-1", "T" => "0.3", "lambda" => "0.0", "cfl" => "0.5", "cutoff" => "1e-12", "maxdim" => "256",
                       "alg" => "densitymatrix", "order" => "seq", "dense" => "1", "check_mpo" => "1", "nsample" => "16",
                       "log_every" => "10", "dense_steps" => "0", "mps" => "1", "analyze" => "0"))
space = opts["space"]
x0 = parse(Float64, opts["x0"]); X = parse(Float64, opts["X"]); wb = parse(Float64, opts["wb"]); iota = parse(Float64, opts["iota"])
T = parse(Float64, opts["T"]); lambda = parse(Float64, opts["lambda"]); cfl = parse(Float64, opts["cfl"])
cutoff = parse(Float64, opts["cutoff"]); maxdim = parse(Int, opts["maxdim"]); alg = opts["alg"]
nshc(L) = (L + 1) * (L + 2) ÷ 2
# ranks / storage of a dense fabric field Y (space dims..., Nmu, Nphi) for several representations
function analyze_field(Y, nbits, nsd)
    Nmu, Nphi = size(Y)[end-1:end]; Ns = length(Y) ÷ (Nmu * Nphi); Ntot = length(Y)
    Y3 = reshape(Y, Ns, Nmu, Nphi)
    sv = svdvals(reshape(Y, Ns, Nmu * Nphi)); tail = sqrt.(reverse(cumsum(reverse(sv .^ 2)))) ./ norm(sv)
    rsep(e) = findfirst(i -> (i == length(sv) ? 0.0 : tail[i+1]) <= e, 1:length(sv))
    mus = mu_grid(Nmu); lmax = min(Nmu - 2, 120)
    lin = round.(Int, range(1, Ns, length = min(Ns, 256)))
    Es = reduce(hcat, [sh_energy_stream(Y3[l, :, :], mus, fill(2 / Nmu, Nmu), 2pi / Nphi, lmax) for l in lin])
    Et = vec(sum(Es, dims = 2)); rel = sqrt.(max.(0.0, reverse(cumsum(reverse(Et)))) ./ sum(Et))
    sd = collect(1:nsd); od = [nsd + 1, nsd + 2]
    @printf("  analysis of the dense final state (%d values); SH from grid quadrature, lmax %d, tail beyond lmax-8: %.1e\n", Ntot, lmax, sum(Et[end-7:end]) / sum(Et))
    for e in (1e-2, 1e-3, 1e-4)
        L = findfirst(l -> iseven(l) && (l + 2 > lmax + 1 ? 0.0 : rel[l+2]) <= e, 0:lmax); L = L === nothing ? -1 : L - 1
        r = rsep(e)
        _, npo, _ = tt_svd(quantics_tensor(Y, nbits, seq_order([od; sd], nbits)), e)
        rko, _, _ = tt_svd(quantics_tensor(Y, nbits, seq_order([od; sd], nbits)), e)
        _, nps, _ = tt_svd(quantics_tensor(Y, nbits, seq_order([sd; od], nbits)), e)
        _, npt, _ = tt_svd(Y, e)
        @printf("  eps %.0e: r_sep %3d | SH L %3d: dense-space SH %10d, SH+SVD %9d | QTT oseq %8d (max rank %d), QTT seq %8d, TT %8d | dense %d\n",
                e, r, L, L >= 0 ? Ns * nshc(L) : -1, L >= 0 ? r * (Ns + nshc(L)) : -1, npo, maximum(rko), nps, npt, Ntot)
        println("RESULT_ANALYZE eps=$e rsep=$r shL=$L sh_dense=$(L >= 0 ? Ns * nshc(L) : -1) sh_svd=$(L >= 0 ? r * (Ns + nshc(L)) : -1) qtt_oseq=$npo qtt_seq=$nps tt=$npt Ntot=$Ntot")
    end
end

function run_case(nz, nx, nmu)
    nphi = nmu + 1; Nmu, Nphi = 2^nmu, 2^nphi
    f = DivideFlow(wb = wb)

    # ---- grid and coefficients
    if space == "z"
        nbits = [nz, nmu, nphi]; sdims = [1]; amu, aphi = 2, 3
        zs = cellc(0.0, 1.0, 2^nz)
        Ks = [Kmat(velgrad(f, x0, z), iota) for z in zs]
        sshape = (2^nz,)
    else
        nbits = [nx, nz, nmu, nphi]; sdims = [1, 2]; amu, aphi = 3, 4
        xs = cellc(0.0, X, 2^nx); zs = cellc(0.0, 1.0, 2^nz)
        Ks = [Kmat(velgrad(f, x, z), iota) for x in xs, z in zs]
        sshape = (2^nx, 2^nz)
    end
    edw = map(Kedw, Ks)
    E = map(first, edw); Dd = map(x -> x[2], edw); W = map(last, edw)
    ord = opts["order"]
    sd = collect(sdims); od = [amu, aphi]
    order = ord == "seq" ? seq_order([sd; od], nbits) :
            ord == "oseq" ? seq_order([od; sd], nbits) :
            ord == "seq_ointer" ? [seq_order(sd, nbits); inter_order(od, nbits)] :
            ord == "sinter_ointer" ? [inter_order(sd, nbits); inter_order(od, nbits)] :
            ord == "inter" ? inter_order([sd; od], nbits) : error("order?")
    g = FGrid(nbits, order)
    Ns = prod(sshape); Ntot = Ns * Nmu * Nphi
    println("space=$space grid $(sshape) x $(Nmu)x$(Nphi) = $Ntot dofs ($(length(order)) sites), order=$ord, T=$T, lambda=$lambda, threads=$(Threads.nthreads())")

    # ---- time step from the advection and diffusion limits of RK4 (imag. axis 2.8, real axis 2.78)
    hmu, hphi = 2 / Nmu, 2pi / Nphi
    vphi = maximum(abs(k[3]) + abs(k[2]) + abs(k[1]) for k in edw)                  # |w| + |d| + |e| >= |V_phi|
    vmu = maximum(abs(k[1]) + abs(k[2]) for k in edw) * 2 / (3sqrt(3))              # |g q| <= max g * (|e| + |d|)
    adv = vphi / hphi + vmu / hmu
    muc1 = mu_grid(Nmu)[end]
    dif = lambda * (4 / hmu^2 + 4 / (hphi^2 * (1 - muc1^2)))
    dt0 = cfl * min(2.8 / adv, dif > 0 ? 2.78 / dif : Inf)
    nsteps = ceil(Int, T / dt0); dt = T / nsteps
    @printf("max |V_phi| %.2f, |V_mu| %.2f, adv rate %.1f, diff rate %.1f -> dt = %.3e, %d steps\n", vphi, vmu, adv, dif, dt, nsteps)

    # ---- fabric strength of the exact solution at T (sample of space points)
    let xq = gauss_legendre(64), pq = phi_grid(128), lin = round.(Int, range(1, length(Ks), length = min(64, length(Ks))))
        a1s = [begin
            M = Mfrom(expK(Ks[l], T)); F = [psi_closed(M, m, p) for m in xq[1], p in pq]
            a2_eigs(F, xq[1], xq[2], pq)[1][1]
        end for l in lin]
        hw = minimum(begin lm = Meigs(Mfrom(expK(K, T))); rad2deg(sqrt(lm[1] / lm[2])) end for K in Ks)
        @printf("exact fabric at T: a1 in [%.3f, %.3f] (sample), sharpest half-width %.2f deg (h_phi = %.2f deg)\n", minimum(a1s), maximum(a1s), hw, 360 / Nphi)
    end

    # ---- operator
    t0 = time()
    A, O = fabric_mpo(g, (E, Dd, W), sdims, amu, aphi, lambda)
    @printf("MPO built in %.1f s: maxlinkdim(A) = %d, (Oe, Od, Ow, Lap) = (%d, %d, %d, %d); A link dims %s\n", time() - t0, maxlinkdim(A),
            maxlinkdim(O.Oe), maxlinkdim(O.Od), maxlinkdim(O.Ow), maxlinkdim(O.Lap), linkdims(A))

    P = DenseFabric(E, Dd, W, lambda, Nmu, Nphi)
    closed(t) = begin
        Ms = map(K -> Mfrom(expK(K, t)), Ks)
        dense_psi(Ms, mu_grid(Nmu), phi_grid(Nphi))
    end
    reldiff(a, b) = norm(a - b) / norm(b)

    # ---- check the MPO against the dense operator on the closed-form state at T/2
    if opts["check_mpo"] == "1" && Ntot <= 2^22
        Y = closed(T / 2)
        y = dense_to_mps(Y, g; eps = 1e-13)
        Ay = mps_to_dense(apply(A, y; alg = "naive", truncate = false), g)
        out = similar(reshape(Y, Ns, Nmu, Nphi)); dense_rhs!(out, reshape(Y, Ns, Nmu, Nphi), P)
        @printf("MPO check: |A_mpo psi - A_dense psi| / |A_dense psi| = %.2e   (state rank %d)\n", reldiff(vec(Ay), vec(out)), maxlinkdim(y))
        # mass conservation of the discrete operator
        @printf("           sum(A psi) / sum|A psi| = %.1e\n", sum(out) / sum(abs, out))
    end

    # ---- MPS run
    if opts["mps"] == "1"
    psi = embed_mps(fill(1 / (4pi), Nmu), [amu], g)
    massf(m) = inner(embed_mps(ones(Nmu), [amu], g), m) * hmu * hphi / Ns
    mass0 = massf(psi)
    logv = parse(Int, opts["log_every"])
    t0 = time(); tlast = t0
    chis = Int[]
    for n in 1:nsteps
        psi = rk4_mps(psi, A, dt; cutoff = cutoff, maxdim = maxdim, alg = alg)
        push!(chis, maxlinkdim(psi))
        if n % logv == 0 || n == nsteps || n <= 2
            @printf("  step %5d/%d  t=%.4f  chi=%3d  mass-1=%.2e  %.3f s/step\n", n, nsteps, n * dt, maxlinkdim(psi), massf(psi) / mass0 - 1,
                    (time() - tlast) / (n <= 2 ? 1 : (n % logv == 0 ? logv : n % logv)))
            tlast = time(); flush(stdout)
        end
    end
    tmps = time() - t0
    @printf("MPS run: %.1f s total, %.3f s/step, final chi %d (max %d), params %d (dense %d, compression %.1f)\n",
            tmps, tmps / nsteps, maxlinkdim(psi), maximum(chis), nparams(psi), Ntot, Ntot / nparams(psi))
    end
    println("RESULT_STEP space=$space nz=$nz nx=$nx nmu=$nmu order=$ord lambda=$lambda cutoff=$cutoff alg=$alg T=$T nsteps=$nsteps mps_s_per_step=$(opts["mps"] == "1" ? tmps / nsteps : NaN) chi=$(opts["mps"] == "1" ? maximum(chis) : 0) params=$(opts["mps"] == "1" ? nparams(psi) : 0) Ntot=$Ntot mpo_chi=$(maxlinkdim(A))")

    # ---- dense run (full or a sample of space points)
    if opts["dense"] == "1" && Ntot <= 2^28
        ds = parse(Int, opts["dense_steps"]); ds = ds == 0 ? nsteps : min(ds, nsteps)
        Y = fill(1 / (4pi), Ns, Nmu, Nphi); buf = (similar(Y), similar(Y))
        rk4_dense!(Y, P, dt, buf)      # compile
        Y .= 1 / (4pi)
        t0 = time()
        for n in 1:ds
            rk4_dense!(Y, P, dt, buf)
        end
        tden = time() - t0
        @printf("dense run (%d steps): %.1f s, %.4f s/step\n", ds, tden, tden / ds)
        println("RESULT_DENSE space=$space nz=$nz nx=$nx nmu=$nmu Ntot=$Ntot dense_s_per_step=$(tden / ds) threads=$(Threads.nthreads())")
        if ds == nsteps
            Yc = closed(T)
            Ym = (Ntot <= 2^24 && opts["mps"] == "1") ? mps_to_dense(psi, g) : nothing
        opts["analyze"] == "1" && analyze_field(reshape(Y, sshape..., Nmu, Nphi), nbits, length(sdims))
            lambda == 0 && @printf("dense vs closed form: rel L2 %.3e, max %.3e (rel to max)\n", reldiff(vec(Y), vec(Yc)), maximum(abs.(vec(Y) - vec(Yc))) / maximum(Yc))
            if Ym !== nothing
                @printf("MPS vs dense: rel L2 %.3e, max %.3e (rel to max)\n", reldiff(vec(Ym), vec(Y)), maximum(abs.(vec(Ym) - vec(Y))) / maximum(Y))
                lambda == 0 && @printf("MPS vs closed form: rel L2 %.3e\n", reldiff(vec(Ym), vec(Yc)))
                # ranks of the dense final state for reference
                for e in (1e-2, 1e-4, 1e-6)
                    rk, np, _ = tt_svd(quantics_tensor(reshape(Y, sshape..., Nmu, Nphi), nbits, order), e)
                    @printf("  dense final state QTT (%s) eps %.0e: max rank %d, params %d\n", ord, e, maximum(rk), np)
                end
            end
        end
    else
        # dense reference on a sample of space points (operator is local in space)
        ns_ = parse(Int, opts["nsample"])
        lin = round.(Int, range(1, Ns, length = ns_))
        Ps = DenseFabric(E[lin], Dd[lin], W[lin], lambda, Nmu, Nphi)
        Y = fill(1 / (4pi), ns_, Nmu, Nphi); buf = (similar(Y), similar(Y))
        t0 = time()
        for n in 1:nsteps
            rk4_dense!(Y, Ps, dt, buf)
        end
        @printf("dense sample run (%d space points): %.1f s -> extrapolated full dense %.3f s/step\n", ns_, time() - t0, (time() - t0) / nsteps * Ns / ns_)
        Ev = MPSEvaluator(psi)
        cart = CartesianIndices(sshape)
        Ym = zeros(ns_, Nmu, Nphi)
        for (p, l) in enumerate(lin), i in 1:Nmu, j in 1:Nphi
            I = Tuple(cart[l])
            Ym[p, i, j] = Ev(fbits(g, (map(x -> x - 1, I)..., i - 1, j - 1)))
        end
        @printf("MPS vs dense on %d sampled space points: rel L2 %.3e\n", ns_, reldiff(vec(Ym), vec(Y)))
        if lambda == 0
            Kl = Ks[lin]
            Yc = dense_psi(map(K -> Mfrom(expK(K, T)), Kl), mu_grid(Nmu), phi_grid(Nphi))
            @printf("dense vs closed form on the sample: rel L2 %.3e; MPS vs closed form %.3e\n", reldiff(vec(Y), vec(Yc)), reldiff(vec(Ym), vec(Yc)))
        end
    end

end

# cases: "nz,nmu;nz,nmu" (space=z) or "nx,nz,nmu;..." (space=xz); default from nz/nx/nmu
cases = get(opts, "cases", space == "z" ? "$(opts["nz"]),$(opts["nmu"])" : "$(opts["nx"]),$(opts["nz"]),$(opts["nmu"])")
for c in split(cases, ";")
    v = parse.(Int, split(c, ","))
    space == "z" ? run_case(v[1], 0, v[2]) : run_case(v[2], v[1], v[3])
    flush(stdout)
end
println("done")
