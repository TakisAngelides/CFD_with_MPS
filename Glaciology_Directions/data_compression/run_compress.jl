# run_compress.jl -- QTT compression experiments on the crops written by prep_data.jl
#
#   julia run_compress.jl field=grl_bed task=rd n=13            rate-distortion at 2^n x 2^n (centred sub-crop)
#   julia run_compress.jl field=grl_bed task=scale n=13 nmin=8  ranks vs resolution (coarsened) and extent (crops)
# options: ords=interleaved,sequential  baselines=1  eps=0.3,0.1,...  coarse=0 (rd on a block-mean
#          coarsened version: 2^coarse x 2^coarse blocks)
#
# Fields (data = Float32 crops in data/):
#   grl_bed      BedMachine Greenland bed, 150 m (defined everywhere incl. bathymetry)
#   grl_thk      ice thickness, 0 outside the ice                         (rms on ice, rms_all on all)
#   grl_thk_ice  thickness, non-ice cells replaced by a harmonic extension (errors on ice cells)
#   grl_srf      surface elevation, 0 over the ocean                       (rms on ice, rms_all on all)
#   grl_srf_ice  surface, non-ice cells harmonically extended              (errors on ice cells)
#   grl_mask     0/1 ice mask
#   ant_speed    MEaSUREs Antarctic speed, 450 m, no-data = 0               (errors on data cells)
#   ant_logspeed log10(max(speed, 0.1 m/a)), no-data harmonically extended (errors on data cells)
#   ant_bed      BedMachine Antarctica v3 bed, 2 km (sparse version)

include("qttlib.jl")
using HDF5

BLAS.set_num_threads(Threads.nthreads())
const ARGS_D = Dict(String(split(a, "=")[1]) => String(split(a, "=")[2]) for a in ARGS)
getarg(k, d) = get(ARGS_D, k, d)

const DATA = joinpath(@__DIR__, "data")
const RES = joinpath(@__DIR__, "results")
mkpath(RES)

struct FieldSpec
    A::Matrix{Float64}         # what is compressed (transformed + filled)
    mask::BitMatrix            # cells on which errors are measured
    back::Function             # transformed -> physical value
    relphys::Bool              # report relative physical errors (speeds)
    targets::Vector{Float64}   # absolute RMS targets (transformed units) for the scaling study
    unit::String
end

function load_field(field::String)
    if startswith(field, "grl_")
        h = h5open(joinpath(DATA, "grl8192.h5"), "r")
        m = read(h["mask"]); ice = BitMatrix((m .== 2) .| (m .== 3))
        if field == "grl_bed"
            A = Float64.(read(h["bed"])); v = BitMatrix(A .!= -9999.0)
            all(v) || (A = harmonic_fill(A, v))
            spec = FieldSpec(A, v, identity, false, [3.0, 10.0, 30.0], "m")
        elseif field == "grl_thk"
            spec = FieldSpec(Float64.(read(h["thickness"])), ice, identity, false, [3.0, 10.0, 30.0], "m")
        elseif field == "grl_thk_ice"
            spec = FieldSpec(harmonic_fill(Float64.(read(h["thickness"])), ice), ice, identity, false, [3.0, 10.0, 30.0], "m")
        elseif field == "grl_srf"
            spec = FieldSpec(Float64.(read(h["surface"])), ice, identity, false, [1.0, 3.0, 10.0], "m")
        elseif field == "grl_srf_ice"
            spec = FieldSpec(harmonic_fill(Float64.(read(h["surface"])), ice), ice, identity, false, [1.0, 3.0, 10.0], "m")
        elseif field == "grl_mask"
            spec = FieldSpec(Float64.(ice), trues(size(ice)), identity, false, [0.03, 0.1, 0.3], "-")
        else
            error("unknown field $field")
        end
        close(h)
        return spec
    elseif field in ("ant_speed", "ant_logspeed")
        s = Float64.(h5read(joinpath(DATA, "antvel8192.h5"), "speed"))
        v = BitMatrix(isfinite.(s))
        if field == "ant_speed"
            return FieldSpec(ifelse.(v, s, 0.0), v, identity, true, [3.0, 10.0, 30.0], "m/a")
        else
            return FieldSpec(harmonic_fill(log10.(max.(ifelse.(v, s, 0.1), 0.1)), v), v, x -> 10.0^x, true, [0.05, 0.1, 0.2], "log10(m/a)")
        end
    elseif field == "ant_bed"
        A = Float64.(h5read(joinpath(DATA, "antbed2048.h5"), "bed"))
        v = BitMatrix(isfinite.(A) .& (A .!= -9999.0))
        all(v) || (A = harmonic_fill(ifelse.(v, A, 0.0), v))
        return FieldSpec(A, v, identity, false, [10.0, 30.0, 100.0], "m")
    end
    error("unknown field $field")
end

# centred 2^n sub-crop, or 2^n block-mean coarsening (mask: majority of the block)
function subcrop(S::FieldSpec, n::Int)
    N0 = size(S.A, 1); N = 2^n; o = (N0 - N) ÷ 2
    I = o+1:o+N
    return FieldSpec(S.A[I, I], S.mask[I, I], S.back, S.relphys, S.targets, S.unit)
end

function coarsen(S::FieldSpec, levels::Int)
    A = S.A; M = Float64.(S.mask)
    for _ in 1:levels
        A = coarsen2(A); M = coarsen2(M)
    end
    return FieldSpec(A, BitMatrix(M .>= 0.5), S.back, S.relphys, S.targets, S.unit)
end

function physerr(S::FieldSpec, B::Matrix{Float64})
    # 0/1 mask: fraction of misclassified cells after thresholding at 1/2
    S.unit == "-" && return (count(k -> (B[k] > 0.5) != (S.A[k] > 0.5), eachindex(B)) / length(B), NaN)
    S.back === identity && !S.relphys && return (NaN, NaN)
    s2 = 0.0; r2 = 0.0; k = 0
    @inbounds for i in eachindex(B)
        S.mask[i] || continue
        t = S.back(S.A[i]); a = S.back(B[i])
        s2 += (a - t)^2; r2 += ((a - t) / max(abs(t), 1e-3))^2; k += 1
    end
    return sqrt(s2 / k), sqrt(r2 / k)
end

const HEADER = "field,n,method,param,params,bytes,ratio,rms,maxerr,rms_all,phys_rms,phys_relrms,maxrank,time_s,ranks"

function row(io, field, n, method, param, params, bytes, S, B, maxrank, t; ranks = "")
    st = errstats(S.A, B, S.mask)
    pr, prel = physerr(S, B)
    line = @sprintf("%s,%d,%s,%.4g,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.2f,%s",
                    field, n, method, param, params, bytes, params / length(B), st.rms, st.maxerr, st.rms_all, pr, prel, maxrank, t, ranks)
    println(io, line); flush(io)
    println(line); flush(stdout)
end

function rd_task(field, S::FieldSpec, n, tag)
    N = 2^n; cells = N^2
    ords = Symbol.(split(getarg("ords", "interleaved,sequential"), ","))
    epss = parse.(Float64, split(getarg("eps", "0.3,0.1,0.03,0.01,0.003,0.001,0.0003,0.0001,0.00003"), ","))
    file = joinpath(RES, "rd_$(field)_$(tag).csv")
    io = open(file, "w"); println(io, HEADER)
    @printf("# %s %s: N=%d, ||A||_rms=%.4g, std=%.4g, mask fraction %.3f\n", field, tag, N, sqrt(sum(abs2, S.A) / cells), std(S.A), mean(S.mask))
    for ord in ords
        T = to_chain(S.A, ord)
        for eps in epss
            t = @elapsed q = ttsvd(T; eps = eps, maxparams = 1.5 * cells)
            if q === nothing
                println("# $ord eps=$eps: exceeded 1.5 x cells, stop"); break
            end
            B = qtt_to_array(q, n, ord)
            # independent point check of the layout
            for _ in 1:3
                i, j = rand(0:N-1), rand(0:N-1)
                abs(qtt_eval(q, i, j, n, ord) - B[i+1, j+1]) < 1e-6 * (1 + abs(B[i+1, j+1])) || error("point check failed")
            end
            r = qtt_ranks(q)
            row(io, field, n, "qtt_" * String(ord)[1:3], eps, qtt_params(q), 4qtt_params(q), S, B, maximum(r), t; ranks = join(r, ";"))
        end
        T = nothing; GC.gc()
    end
    getarg("baselines", "1") == "1" || (close(io); return)

    # matrix truncated SVD
    t = @elapsed F = svd(S.A)
    rs = unique(clamp.(round.(Int, 2.0 .^ (0:0.5:n)), 1, N))
    for r in rs
        t2 = @elapsed B = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
        row(io, field, n, "svd", r, r * (2N + 1), 4r * (2N + 1), S, B, r, t + t2)
    end
    F = nothing; GC.gc()

    # wavelets: keep the K largest coefficients
    levels = n - 3
    fracs = 10.0 .^ (-(0.25:0.25:4.5))
    for (name, f, fi) in (("haar", haar_fwd!, haar_inv!), ("cdf97", cdf97_fwd!, cdf97_inv!))
        t = @elapsed C = wt2d!(copy(S.A), levels, f)
        sa = sort(abs.(vec(C)), rev = true)
        for fr in fracs
            K = max(1, round(Int, fr * cells))
            t2 = @elapsed B = iwt2d!(keep_largest(C, K, sa), levels, fi)
            row(io, field, n, name, fr, K, round(Int, 4K + sigmap_bytes(K, cells)), S, B, 0, t + t2)
        end
        C = nothing; sa = nothing; GC.gc()
    end

    # quantisation + zlib (bytes reported; params = bytes/4 for the Float32-equivalent count)
    sref = std(S.A[S.mask])
    for x in 0:0.5:5
        Δ = sref * 10.0^(-x - 1)
        t = @elapsed (b, B, Ti) = quant_zlib(S.A, Δ)
        row(io, field, n, "zlib", Δ, cld(b, 4), b, S, B, 0, t)
    end
    # lossless zlib of the Float32 data for reference
    b = zlib_bytes(collect(reinterpret(UInt8, vec(Float32.(S.A)))))
    row(io, field, n, "zlib_f32_lossless", 0.0, cld(b, 4), b, S, Float64.(Float32.(S.A)), 0, 0.0)
    close(io)
end

function scale_task(field, S0::FieldSpec, n, nmin)
    ords = Symbol.(split(getarg("ords", "interleaved,sequential"), ","))
    kinds = split(getarg("kinds", "coarse,crop"), ",")
    file = joinpath(RES, "scale_$(field)_n$(n).csv")
    io = open(file, "w")
    println(io, "field,kind,m,method,target,eps,maxrank,params,ratio,rms_all,rms,maxerr,time_s,ranks")
    for kind in kinds, m in nmin:n
        S = kind == "coarse" ? coarsen(S0, n - m) : subcrop(S0, m)
        N = 2^m; cells = N^2
        nrm = norm(S.A)
        # matrix-SVD rank for the same absolute target (all cells)
        sv = svdvals(S.A)
        tail = reverse(cumsum(reverse(sv .^ 2)))
        for target in S.targets
            rsv = something(findfirst(r -> (r < N ? tail[r+1] : 0.0) <= target^2 * cells, 1:N), N)
            line = @sprintf("%s,%s,%d,svd,%.4g,%.4g,%d,%d,%.6g,%.6g,,,0,", field, kind, m, target, target * N / nrm, rsv, rsv * (2N + 1), rsv * (2N + 1) / cells, sqrt((rsv < N ? tail[rsv+1] : 0.0) / cells))
            println(io, line); println(line)
        end
        for ord in ords
            T = to_chain(S.A, ord)
            for target in S.targets
                eps = target * N / nrm
                t = @elapsed q = ttsvd(T; eps = eps, maxparams = 1.5 * cells)
                if q === nothing
                    line = @sprintf("%s,%s,%d,qtt_%s,%.4g,%.4g,-1,-1,-1,,,,%.2f,", field, kind, m, String(ord)[1:3], target, eps, t)
                    println(io, line); println(line); continue
                end
                B = qtt_to_array(q, m, ord)
                st = errstats(S.A, B, S.mask)
                r = qtt_ranks(q)
                line = @sprintf("%s,%s,%d,qtt_%s,%.4g,%.4g,%d,%d,%.6g,%.6g,%.6g,%.6g,%.2f,%s", field, kind, m, String(ord)[1:3], target, eps,
                                maximum(r), qtt_params(q), qtt_params(q) / cells, st.rms_all, st.rms, st.maxerr, t, join(r, ";"))
                println(io, line); println(line); flush(io); flush(stdout)
            end
        end
    end
    close(io)
end


# QTT by tensor cross interpolation from point samples of the array (no full SVDs; counts samples)
import TensorCrossInterpolation as TCI
function tci_task(field, S::FieldSpec, n)
    N = 2^n; cells = N^2
    ords = Symbol.(split(getarg("ords", "interleaved"), ","))
    tols = parse.(Float64, split(getarg("tols", "0.01,0.003,0.001,0.0003"), ","))
    maxD = parse(Int, getarg("maxbonddim", "400"))
    file = joinpath(RES, "tci_$(field)_n$(n).csv")
    io = open(file, "w")
    println(io, "field,n,method,tol,evals,unique_evals,unique_frac,maxrank,params,ratio,rms,maxerr,rms_all,time_s,ranks")
    for ord in ords
        px, py = site_positions(n, ord)
        for tol in tols
            calls = Ref(0); seen = Set{Int}()
            f = sigma -> begin
                i = 0; j = 0
                for k in 1:n
                    i = 2i + (sigma[px[k]] - 1); j = 2j + (sigma[py[k]] - 1)
                end
                calls[] += 1; push!(seen, i + N * j)
                S.A[i+1, j+1]
            end
            # start from the largest |value| on a coarse lattice
            sub = round.(Int, range(0, N - 1, length = 64))
            best = argmax(((i, j),) -> abs(S.A[i+1, j+1]), [(i, j) for i in sub, j in sub])
            piv = zeros(Int, 2n)
            for k in 1:n
                piv[px[k]] = ((best[1] >> (n - k)) & 1) + 1; piv[py[k]] = ((best[2] >> (n - k)) & 1) + 1
            end
            t = @elapsed tci, ranks, errors = TCI.crossinterpolate2(Float64, f, fill(2, 2n), [piv]; tolerance = tol, maxbonddim = maxD)
            tt = TCI.TensorTrain(tci)
            q = QTT([Array{Float64,3}(c) for c in tt.sitetensors])
            B = qtt_to_array(q, n, ord)
            st = errstats(S.A, B, S.mask)
            r = qtt_ranks(q)
            line = @sprintf("%s,%d,tci_%s,%.3g,%d,%d,%.5g,%d,%d,%.6g,%.6g,%.6g,%.6g,%.1f,%s", field, n, String(ord)[1:3], tol, calls[], length(seen),
                            length(seen) / cells, maximum(r), qtt_params(q), qtt_params(q) / cells, st.rms, st.maxerr, st.rms_all, t, join(r, ";"))
            println(io, line); println(line); flush(io); flush(stdout)
        end
    end
    close(io)
end

field = getarg("field", "grl_bed")
task = getarg("task", "rd")
n = parse(Int, getarg("n", "13"))
t = @elapsed S0 = load_field(field)
@printf("# loaded %s (%d^2) in %.1f s, threads %d\n", field, size(S0.A, 1), t, Threads.nthreads())
n0 = round(Int, log2(size(S0.A, 1)))
if task == "rd"
    c = parse(Int, getarg("coarse", "0"))
    S = c > 0 ? coarsen(S0, c) : S0
    n1 = n0 - c
    S = n < n1 ? subcrop(S, n) : S
    rd_task(field, S, min(n, n1), c > 0 ? "c$(c)_n$(min(n, n1))" : "n$(min(n, n1))")
elseif task == "scale"
    scale_task(field, n < n0 ? subcrop(S0, n) : S0, min(n, n0), parse(Int, getarg("nmin", "8")))
elseif task == "tci"
    tci_task(field, n < n0 ? subcrop(S0, n) : S0, min(n, n0))
end
println("done")
