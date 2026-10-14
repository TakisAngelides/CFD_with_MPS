# Task 2: quantics ranks of the isostasy Fourier multipliers (TCI, rev bit order) vs resolution, and
# ranks of loads / transformed loads / responses (dense SVD) to size the MPS pipeline.
# usage: julia t2_ranks.jl [part=AB C] [nmaxA=20] [nmaxB=11]
include("isostasy.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "OPENBLAS_NUM_THREADS", "1"))); FFTW.set_num_threads(1)

opts = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS)
parts = get(opts, "part", "ABC")
nmaxA = parse(Int, get(opts, "nmaxA", "20"))
nmaxB = parse(Int, get(opts, "nmaxB", "11"))
p = Earth()
W = 3000e3; L = 2W
@printf("Earth: D = %.3e N m, flexural length %.1f km, eta %.1e Pa s\n", rigidity(p), flexural_length(p) / 1e3, p.eta)

chi_at(psi, c) = maxlinkdim(truncate!(copy(psi); cutoff = c))

mults = [("eq (1/beta)", k -> mult_eq(k, p)),
         ("resp t=1 kyr", k -> mult_resp(k, 1000.0, p)),
         ("decay dt=10 yr", k -> mult_decay(k, 10.0, p)),
         ("decay dt=100 yr", k -> mult_decay(k, 100.0, p)),
         ("Boussinesq 1/|k|", k -> mult_bouss(k))]

if occursin("A", parts)
    println("\n== A. Fourier multipliers: TCI (tol 1e-13) then SVD recompression; 2D, L = 6000 km ==")
    for ord in (:interleaved, :sequential)
        println("  ordering $ord")
        @printf("  %-18s %3s %8s %8s %8s %8s %8s %9s %10s\n", "multiplier", "n", "chi_TCI", "eps=1e-4", "eps=1e-6", "eps=1e-8", "eps=1e-12", "t_TCI", "maxerr")
        for (name, f) in mults, n in sort(unique([collect(6:2:nmaxA); 30]))
            g = QGrid(2, n; ordering = ord)
            N = npts(g)
            h(idx) = f(kabs(idx, N, L))
            (res, t) = timed(() -> tci_mps(h, g; rev = true, tolerance = 1e-13, maxbonddim = 400))
            psi, rt, et = res
            # accuracy: dense check for n <= 9, else random sample
            E = MPSEval(psi)
            if n <= 9
                errs = [abs(evaluate(E, g, (i, j); rev = true) - h((i, j))) for i in 0:N-1, j in 0:N-1]
                me = maximum(errs)
            else
                # half the samples near k = 0 (signed indices |m| < 512, where the multipliers live), half uniform
                sm(m) = mod(m, N)
                pts = [[(sm(rand(-512:512)), sm(rand(-512:512))) for _ in 1:20000]; [(rand(0:N-1), rand(0:N-1)) for _ in 1:20000]]
                me = maximum(abs(evaluate(E, g, ij; rev = true) - h(ij)) for ij in pts)
            end
            scale = maximum(abs(h((i, j))) for i in 0:min(N - 1, 40), j in 0:min(N - 1, 40))
            @printf("  %-18s %3d %8d %8d %8d %8d %8d %8.2fs %10.1e\n", name, n, rt, chi_at(psi, 1e-8), chi_at(psi, 1e-12), chi_at(psi, 1e-16), chi_at(psi, 1e-24), t, me / scale)
        end
    end
end

# cutoff c (discarded squared weight) <-> relative L2 truncation error ~ sqrt(c)
function field_ranks(name, A, g; rev = false, cuts = (1e-4, 1e-8, 1e-12, 1e-16))
    (psi, t) = timed(() -> array_to_mps(A, g; rev = rev, cutoff = minimum(cuts) * 1e-4))
    chis = [chi_at(psi, c) for c in cuts]
    @printf("  %-34s n=%2d %-11s  chi: %s   (svd %.1fs)\n", name, g.n, g.ordering, join([@sprintf("%5d", c) for c in chis], " "), t)
    return chis
end

if occursin("B", parts)
    println("\n== B. Disc loads: ranks of sigma, sigma^ (rev), u(t=1 kyr), u_eq; rel. errors 1e-2 1e-4 1e-6 1e-8 ==")
    for n in 6:nmaxB, ord in (:interleaved, :sequential)
        g = QGrid(2, n; ordering = ord); N = npts(g)
        x = cellcentres(N, W)
        K = kgrid2(N, L)
        for (lname, load) in (("sharp disc", disc_sharp), ("tanh disc w=20km", (a, b) -> disc_smooth(a, b; w = 20e3)),
                              ("tanh disc w=50km", (a, b) -> disc_smooth(a, b; w = 50e3)))
            H = [load(a, b) for a in x, b in x]
            sig = p.rho_ice * p.g .* H
            field_ranks("$lname: H", H, g)
            sh = fft(complex.(sig))
            field_ranks("$lname: sigma^ (rev)", sh, g; rev = true)
            u1 = real.(ifft(sh .* mult_resp.(K, 1000.0, Ref(p))))
            field_ranks("$lname: u(1 kyr)", u1, g)
            ue = real.(ifft(sh .* mult_eq.(K, Ref(p))))
            field_ranks("$lname: u_eq", ue, g)
        end
    end
end

if occursin("C", parts)
    println("\n== C. Antarctic grounded-ice load (BedMachine, 8 km, padded to 1024^2 = 8192 km); rel. errors 1e-2 1e-3 1e-4 1e-6 1e-8 ==")
    H, Wa = antarctic_load()
    La = 2Wa
    @printf("  max H %.0f m, ice volume %.2e km^3\n", maximum(H), sum(H) * 8e3^2 / 1e9)
    for m in (-2, -1, 0), ord in (:interleaved, :sequential)
        # coarser versions by 2^|m| block averaging
        Hm = m == 0 ? H : (b = 2^(-m); [sum(H[(i-1)*b+1:i*b, (j-1)*b+1:j*b]) / b^2 for i in 1:1024÷b, j in 1:1024÷b])
        Nm = size(Hm, 1); n = round(Int, log2(Nm))
        g = QGrid(2, n; ordering = ord)
        K = kgrid2(Nm, La)
        sig = p.rho_ice * p.g .* Hm
        field_ranks("ANT: H", Hm, g; cuts = (1e-4, 1e-6, 1e-8, 1e-12, 1e-16))
        sh = fft(complex.(sig))
        field_ranks("ANT: sigma^ (rev)", sh, g; rev = true, cuts = (1e-4, 1e-6, 1e-8, 1e-12, 1e-16))
        for t in (1000.0, 10000.0)
            u = real.(ifft(sh .* mult_resp.(K, t, Ref(p))))
            field_ranks("ANT: u($(Int(t)) yr)", u, g; cuts = (1e-4, 1e-6, 1e-8, 1e-12, 1e-16))
        end
        ue = real.(ifft(sh .* mult_eq.(K, Ref(p))))
        field_ranks("ANT: u_eq", ue, g; cuts = (1e-4, 1e-6, 1e-8, 1e-12, 1e-16))
        @printf("  (u_eq range %.1f .. %.1f m)\n", extrema(ue)...)
    end
    println("\n  real-space kernels (Green's functions, first column of the circulant), 1024^2:")
    g = QGrid(2, 10)
    K = kgrid2(1024, La)
    for (name, f) in mults
        G = real.(ifft(f.(K)))
        field_ranks("kernel $name", G, g)
    end
end
println("done")
