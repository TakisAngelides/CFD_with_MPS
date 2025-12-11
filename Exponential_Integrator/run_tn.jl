t_total_initial = time()
using ITensors
using Plots
using QuanticsTCI
using TCIITensorConversion
using LinearAlgebra
using HDF5
using Dates
include("utilities_common.jl")
include("utilities_tn.jl")

# Hadamard gate for ITensors
println("Now getting the hadamard ", now())
flush(stdout)
h = [1 1; 1 -1] # hadamard
ITensors.op(::OpName"h", ::SiteType"S=1/2") = h # hadamard as ITensor
println("Finished getting the hadamard ", now())
flush(stdout)

# Input arguments for file and opening the results h5 file
println("Now getting the input arguments for the file and opening the results h5 file ", now())
flush(stdout)
path_to_project_number = ARGS[1] 
path_to_inputs_h5 = "$(path_to_project_number)/inputs.h5" # Where the h5 file containing the inputs is
job_id = ARGS[2] # The job id to get the inputs within this h5 file for this specific job
path_to_results = "$(path_to_project_number)/HDF5/$(job_id).h5" # The path to the h5 file to save all results of states
path_to_backup = "$(path_to_project_number)/HDF5_b/$(job_id)_b.h5"
h5open(path_to_results, "w") do results_file
rerun_mode = isfile(path_to_backup)
println("Finished getting the input arguments for the file and opening the results h5 file ", now())
flush(stdout)

# Build a dictionary of the inputs
println("Now building a dictionary of the inputs", now())
flush(stdout)
inputs = Dict()
h5open(path_to_inputs_h5, "r") do inputs_file
    group = inputs_file["$(job_id)"]
    group_attributes = attributes(group)
    for key in keys(group_attributes)
        inputs[key] = read(group_attributes[key])
    end
end
println("The inputs are ", inputs)
flush(stdout)
println("Finished building a dictionary of the inputs ", now())
flush(stdout)

# Parameters
println("Now reading the parameters ", now())
flush(stdout)
x_start = inputs["x_start"] 
x_end = inputs["x_end"]
n_qubits = inputs["n_qubits"]
N = 2^n_qubits
dx = (x_end - x_start)/N
x_values_u0 = x_start .+ dx .* collect(0:N-1) # grid of x
inputs["x_values_u0"] = x_values_u0
tol_ttc_u0 = inputs["tol_ttc_u0"]
qft_mpo_cutoff = inputs["qft_mpo_cutoff"]
iterations = inputs["iterations"]
cutoff = inputs["cutoff"]
qft_sign_convention = inputs["qft_sign_convention"] # set to -1
max_bond_ttc_kx = inputs["max_bond_ttc_kx"]
tol_ttc_kx = inputs["tol_ttc_kx"]
dt = inputs["dt"]
kappa = inputs["kappa"]
delta_val = inputs["delta_val"]
scheme = inputs["scheme"]
which_equation = inputs["which_equation"]
println("Finished reading the parameters ", now())
flush(stdout)

if rerun_mode # Case when the script already ran at least once and saved a backup file but somehow stopped before finishing all the time steps so its rerun
    println("Now reading in from backup file ", now())
    flush(stdout)
    h5open(path_to_backup, "r") do f
        starting_iteration = read(f, "starting_iteration") # Where we left off last time the script stopped running
        uhat0 = read(f, "uhat0", MPS) # This is uhat at starting_iteration
        sites = siteinds(uhat0)
        qft_mpo = read(f, "qft_mpo", MPO)
        qft_inv_mpo = read(f, "qft_inv_mpo", MPO)
        kx_mpo = read(f, "kx_mpo", MPO)
        prop_mpo = read(f, "prop_mpo", MPO)
    end
    println("Finished reading in from backup file ", now())
    flush(stdout)
else
    # Initial state u in real space
    println("Now getting the initial state ", now())
    flush(stdout)
    sites, u0 = get_initial_state(inputs)
    starting_iteration = 1
    println("Finished getting the initial state ", now())
    flush(stdout)

    # Fourier transform mpo and inverse fourier transform mpo 
    println("Now getting the fourier transform mpos ", now())
    flush(stdout)
    qft_mpo = sqrt(1/2^n_qubits)*get_qft_mpo(sites, qft_sign_convention, qft_mpo_cutoff)
    qft_inv_mpo = sqrt(1/2^n_qubits)*get_inv_qft_mpo(sites, qft_sign_convention, cutoff)
    println("Now getting the fourier transform mpos ", now())
    flush(stdout)

    # Getting kx_mpo
    println("Now getting the kx mpo ", now())
    flush(stdout)
    kx_positive = LinRange(0,round(N/2), Int(N/2)+1)*2*pi/(x_end - x_start) # grid of frequencies
    kx_negative = -kx_positive[2:(end-1)] # one less freq negative than positive
    kx = [kx_positive; reverse(kx_negative)] # as they appear after the fft
    sites_kx_mps, kx_mps = tt_cross(x -> kx[Int(x)], 1:length(kx), max_bond_ttc_kx, tol_ttc_kx)
    kx_mps = reverse_mps(kx_mps, sites_kx_mps)
    for i in eachindex(kx_mps)
        existing = sites_kx_mps[end-i+1]
        new = sites[i]
        replaceind!(kx_mps[i], existing => new)
    end
    kx_mpo = get_mpo_out_of_mps(kx_mps)
    println("Finished getting the kx mpo ", now())
    flush(stdout)

    # Getting prop_mpo
    println("Now getting the prop mpo ", now())
    flush(stdout)
    kx2 = kx.^2
    prop_vec = get_prop(kappa, dt, kx2, delta_val, which_equation, inputs)
    sites_prop_mps, prop_mps = tt_cross(x -> prop_vec[Int(x)], 1:length(prop_vec), max_bond_ttc_kx, tol_ttc_kx)
    prop_mps = reverse_mps(prop_mps, sites_prop_mps)
    for i in eachindex(prop_mps)
        existing = sites_prop_mps[end-i+1]
        new = sites[i]
        replaceind!(prop_mps[i], existing => new)
    end
    prop_mpo = get_mpo_out_of_mps(prop_mps)
    println("Finished getting the prop mpo ", now())
    flush(stdout)

    # Initial state uhat in fourier space
    println("Now getting the initial state in fourier space ", now())
    flush(stdout)
    uhat0 = reverse_mps(apply(qft_mpo, u0; cutoff = cutoff), sites)
    println("Finished getting the initial state in fourier space ", now())
    flush(stdout)
end

# Peform evolution
println("Now performing the whole time evolution ", now())
flush(stdout)
if scheme == "euler"
    time_evolution_euler(uhat0, starting_iteration, iterations, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, scheme, results_file, inputs, path_to_backup, t_total_initial)
else
    time_evolution_heun(uhat0, starting_iteration, iterations, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, scheme, results_file, inputs, path_to_backup, t_total_initial)
end
println("Finished performing the whole time evolution ", now())
flush(stdout)

end # close the results file

println("Finished. ", now())
flush(stdout)
