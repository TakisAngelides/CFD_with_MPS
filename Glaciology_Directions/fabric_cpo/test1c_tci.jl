# Test 1c: resolution scan of the 4D fabric field psi(x, z, mu, phi) beyond dense sizes. The QTT is
# built by tensor cross interpolation directly from the closed form (G precomputed on the space grid),
# then SVD-recompressed to the target accuracy. Accuracy is checked at random points.
#   sbatch slurm/run.sbatch test1c_tci.jl cases="5,7;6,7;7,7;8,8" wb=0.35
using Random
include("fabric_mps.jl")

opts = parse_opts(Dict("cases" => "5,6", "wb" => "0.35", "X" => "4.0", "iota" => "-1", "eps" => "1e-2,1e-4",
                       "tol" => "1e-7", "maxbond" => "400", "order" => "oseq", "nsamp" => "200000", "fs" => "0.0"))
wb = parse(Float64, opts["wb"]); X = parse(Float64, opts["X"]); iota = parse(Float64, opts["iota"]); fs = parse(Float64, opts["fs"])
epss = parse.(Float64, split(opts["eps"], ",")); tol = parse(Float64, opts["tol"]); maxbond = parse(Int, opts["maxbond"])
nsamp = parse(Int, opts["nsamp"])
f = DivideFlow(wb = wb, fs = fs)
println("threads $(Threads.nthreads()), wb=$wb X=$X tol=$tol order=$(opts["order"])")

for c in split(opts["cases"], ";")
    ns, nmu = parse.(Int, split(c, ","))
    nphi = nmu + 1
    Nx = Nz = 2^ns; Nmu, Nphi = 2^nmu, 2^nphi
    xs = cellc(0.0, X, Nx); zs = cellc(0.0, 1.0, Nz)
    t0 = time()
    Ms = Matrix{NTuple{3,Float64}}(undef, Nx, Nz)
    Threads.@threads for j in 1:Nz
        for i in 1:Nx
            Ms[i, j] = Mfrom(path_G(f, xs[i], zs[j]; iota = iota))
        end
    end
    tG = time() - t0
    mus = mu_grid(Nmu); phis = phi_grid(Nphi)
    nbits = [ns, ns, nmu, nphi]
    order = opts["order"] == "oseq" ? seq_order([3, 4, 1, 2], nbits) :
            opts["order"] == "seq" ? seq_order([1, 2, 3, 4], nbits) :
            opts["order"] == "ointer_sinter" ? [inter_order([3, 4], nbits); inter_order([1, 2], nbits)] : error("order")
    g = FGrid(nbits, order)
    idx_of(sigma) = begin
        id = zeros(Int, 4)
        for (j, (a, k)) in enumerate(order)
            id[a] |= (sigma[j] - 1) << (nbits[a] - k)
        end
        id
    end
    F(sigma) = (id = idx_of(sigma); psi_closed(Ms[id[1]+1, id[2]+1], mus[id[3]+1], phis[id[4]+1]))
    # pivot at the global maximum (strongest fabric, its peak direction)
    lm = map(Meigs, Ms); I = argmin(first.(lm)); M = Ms[I]
    ev = eigen(Symmetric([M[1] M[2]; M[2] M[3]]))
    v = ev.vectors[:, 1]; php = mod(atan(v[2], v[1]), 2pi)
    ip = clamp(round(Int, php / (2pi / Nphi) - 0.5), 0, Nphi - 1); im = Nmu ÷ 2
    piv = fbits(g, (I[1] - 1, I[2] - 1, im, ip))
    t0 = time()
    tci, ranks, errors = TCI.crossinterpolate2(Float64, F, fill(2, length(order)), [piv]; tolerance = tol, maxbonddim = maxbond, verbosity = 0)
    ttci = time() - t0
    psi = MPS(tci; sites = g.sites)
    @printf("\n=== ns=%d nmu=%d (%d x %d x %d x %d = 2^%d values): G %.1f s, TCI %.1f s, %d sweeps, TCI max rank %d, last err %.1e\n",
            ns, nmu, Nx, Nz, Nmu, Nphi, sum(nbits), tG, ttci, length(errors), maxlinkdim(psi), errors[end])
    rng = MersenneTwister(2)
    samp = [(rand(rng, 0:Nx-1), rand(rng, 0:Nz-1), rand(rng, 0:Nmu-1), rand(rng, 0:Nphi-1)) for _ in 1:nsamp]
    exact = [psi_closed(Ms[s[1]+1, s[2]+1], mus[s[3]+1], phis[s[4]+1]) for s in samp]
    for e in epss
        p2 = truncate!(copy(psi); cutoff = (e / sqrt(length(order) - 1))^2)
        Ev = MPSEvaluator(p2)
        approx = [Ev(fbits(g, s)) for s in samp]
        @printf("  eps %.0e: max rank %3d, params %8d (dense 2^%d = %.2e, compression %.1f); sampled rel L2 err %.2e, max err / max psi %.2e\n",
                e, maxlinkdim(p2), nparams(p2), sum(nbits), 2.0^sum(nbits), 2.0^sum(nbits) / nparams(p2),
                norm(approx - exact) / norm(exact), maximum(abs.(approx - exact)) / maximum(exact))
        println("RESULT_TCI ns=$ns nmu=$nmu wb=$wb order=$(opts["order"]) eps=$e rank=$(maxlinkdim(p2)) params=$(nparams(p2)) links=", join(linkdims(p2), ","))
    end
end
println("done")
