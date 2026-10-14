# Bond dimensions of the SIA fields and operators (Halfar at t0) vs resolution, ordering and cutoff
#   sbatch slurm/run.sbatch bench_ranks.jl
using Printf
include("quantics.jl"); include("sia.jl"); include("benchmarks.jl")
p = SIAParams(); dt = 100.0
tci_tol = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-8          # relative TCI tolerance of H
dims = length(ARGS) >= 2 ? [parse(Int, ARGS[2])] : [1, 2]
println("TCI tolerance ", tci_tol)
println("   d  bits  ordering      field_cutoff  chi(H)  chi(D_x)  chi(L)  chi(A)   t_fields  t_op")
for (d, bitsrange) in filter(t -> t[1] in dims, [(1, 10:2:20), (2, 6:10)])
    hb = Halfar(d = d, p = p)
    for nbits in bitsrange, ordering in (d == 1 ? (:interleaved,) : (:interleaved, :sequential)), fc in (1e-13, 1e-10)
        g = QuanticsGrid(nbits, ordering, ntuple(_ -> (-1200e3, 1200e3), d)...)
        ops = SIAOperators(g)
        st = SolverSettings(cutoff = fc, maxdim = 1024, op_maxdim = 4096)
        H = function_to_mps((x...) -> halfar_H(hb, halfar_t0(hb), sqrt(sum(abs2, x))), g; tolerance = tci_tol, maxbonddim = 512)
        tf = @elapsed D = face_diffusivities(H, nothing, ops, p, st)
        to = @elapsed begin
            L = diffusion_mpo(D, ops, st)
            A = mpo_lincomb([1.0, -dt], [ops.I, L]; cutoff = st.op_cutoff)
        end
        println(@sprintf("  %2d  %4d  %-12s  %.0e       %4d    %4d     %4d    %4d   %7.1f  %6.1f", d, nbits, ordering, fc,
                         maxlinkdim(H), maxlinkdim(D[1]), maxlinkdim(L), maxlinkdim(A), tf, to))
        flush(stdout)
    end
end
