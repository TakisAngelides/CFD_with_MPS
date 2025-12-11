t_total_initial = time()
using LinearAlgebra
using HDF5
using Dates
using FFTW
include("utilities_common.jl")
include("utilities_no_tn.jl")

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
iterations = inputs["iterations"]
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
        uhat0 = read(f, "uhat0") # This is uhat at starting_iteration
    end
    println("Finished reading in from backup file ", now())
    flush(stdout)
else
    # Initial state u in real space
    println("Now getting the initial state ", now())
    flush(stdout)
    u0 = get_initial_state(inputs)
    starting_iteration = 1
    println("Finished getting the initial state ", now())
    flush(stdout)

    # Getting kx
    println("Now getting the kx ", now())
    flush(stdout)
    kx = LinRange(0, div(N, 2), div(N, 2)+1)*2*pi/(x_end - x_start) # grid of frequencies
    println("Finished getting the kx ", now())
    flush(stdout)

    # Getting prop
    println("Now getting the prop ", now())
    flush(stdout)
    kx2 = kx.^2
    prop_vec = get_prop(kappa, dt, kx2, delta_val, which_equation, inputs)
    println("Finished getting the prop ", now())
    flush(stdout)

    # Initial state uhat in fourier space
    println("Now getting the initial state in fourier space ", now())
    flush(stdout)
    uhat0 = rfft(u0)
    println("Finished getting the initial state in fourier space ", now())
    flush(stdout)
end

# Peform evolution
println("Now performing the whole time evolution ", now())
flush(stdout)
if scheme == "euler"
    time_evolution_euler(uhat0, starting_iteration, iterations, dt, kx, prop_vec, scheme, results_file, inputs, path_to_backup, t_total_initial, N)
else
    time_evolution_heun(uhat0, starting_iteration, iterations, dt, kx, prop_vec, scheme, results_file, inputs, path_to_backup, t_total_initial, N)
end
println("Finished performing the whole time evolution ", now())
flush(stdout)

end # close the results file

println("Finished.", now())
flush(stdout)
