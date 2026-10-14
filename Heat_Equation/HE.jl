using ITensors
using Statistics
using Plots
using HDF5
using ITensorMPS
using QuanticsTCI
using TCIITensorConversion
using Random
include("utilities.jl")
include("mpo_inverse.jl")
ITensors.disable_warn_order()

# Physical parameters
N = 20
L_start = -7.0
L_end = 7.0
dx = (L_end - L_start) / (2^N - 1)  
nu = 0.1
dt = 0.01 # 0.1 * dx^2 / nu                      
steps = 500
cutoff = 1e-12
cutoff_inverse = 1e-12
max_sweeps = 5
tol = 1e-9
maxbonddim = 100
tolerance = 1e-12
fixed_boundary_conditions = false
pbc = !fixed_boundary_conditions
plot_every = div(steps, 5)
print_details = false

println("dx = ", dx)
println("dt = ", dt)
println("Total physical time = ", steps * dt)
println("Expected spreading time = ", 0.1^2 / nu)  # width^2 / nu

# TT cross on fine grid for good compression, physics uses dx above
x_values = range(L_start, L_end, length = 2^N)

Random.seed!(1234)

# Narrow Gaussian — will visibly spread under diffusion
function tmp_fun(x)
    return exp(-(x^2) / 0.1)
end
sites, mps = tt_cross(tmp_fun, x_values, maxbonddim, tolerance)

println("Initial linkdims: ", linkdims(mps))

# Sanity check: plot initial condition
x_plot = range(L_start, L_end, length = 500)
function mps_to_list_efficient(mps, sites, num_samples; reverse_flag = false)
    x_inds = []
    y = []
    step = (2^length(mps) - 1) / (num_samples - 1)
    samples = (Int(round(i * step)) for i in 0:num_samples-1)
    for decimal in samples
        state = digits(decimal; base=2, pad=N) .+ 1
        if reverse_flag
            tmp = MPS(sites, state)
        else
            tmp = MPS(sites, reverse(state))
        end
        push!(x_inds, decimal + 1)
        push!(y, inner(mps, tmp))
    end
    return x_inds, y
end

_, y_init = mps_to_list_efficient(mps, sites, 500)
display(Plots.plot(x_plot, real(y_init), title="Initial condition", label="t=0"))

# Fixed boundary conditions (not used here but kept for generality)
if fixed_boundary_conditions
    fixed_boundary_conditions_mpo = get_boundary_conditions_MPO(sites)
    mps = apply(fixed_boundary_conditions_mpo, mps; cutoff = cutoff)
end

# Precompute inverse of B = 25*I - 12*dt*nu*L
function get_inverse_of_B(sites, dt, dx, cutoff_inverse, max_sweeps, tol, nu, pbc)
    N = length(sites)
    res_opsum = OpSum()
    res_opsum += 25, "Id", 1
    L_opsum = sr1(N, pbc) / (dx^2)
    L_opsum -= 2 / (dx^2), "Id", 1
    L_opsum += sl1(N, pbc) / (dx^2)
    res_opsum -= nu * 12 * dt * L_opsum
    mpo = MPO(res_opsum, sites)
    res = get_inverse(mpo, sites, cutoff_inverse, max_sweeps, tol)
    return res
end

println("Computing MPO inverse...")
mpo_A = get_inverse_of_B(sites, dt, dx, cutoff_inverse, max_sweeps, tol, nu, pbc)
println("MPO inverse done. linkdims: ", linkdims(mpo_A))

# Derivative MPO — not needed for pure heat equation but kept for generality
function get_derivative_mpo(N, sites, dx)
    opsum = sr1(N) / (2 * dx)
    opsum -= sl1(N) / (2 * dx)
    return MPO(opsum, sites)
end
derivative_mpo = get_derivative_mpo(N, sites, dx)

# No nonlinear term for heat equation
function get_N(mps_list, t_shift, derivative_mpo)
    return 0.0 * mps_list[end]  # zero MPS
end

# Time evolution
mps_list = [mps, mps, mps, mps]

function time_evolution(steps, mps_list, mpo_A, dt, derivative_mpo)
    mps = mps_list[end]
    D_list = [linkdims(mps), linkdims(mps), linkdims(mps), linkdims(mps)]
    for step in 1:steps
        rhs = 48 * mps_list[end] - 36 * mps_list[end-1] + 16 * mps_list[end-2] - 3 * mps_list[end-3] +
              48 * dt * get_N(mps_list, 0, derivative_mpo) -
              72 * dt * get_N(mps_list, 1, derivative_mpo) +
              48 * dt * get_N(mps_list, 2, derivative_mpo) -
              12 * dt * get_N(mps_list, 3, derivative_mpo)
        mps = apply(mpo_A, rhs; cutoff = cutoff)
        if fixed_boundary_conditions
            mps = apply(fixed_boundary_conditions_mpo, mps; cutoff = cutoff)
        end
        push!(mps_list, mps)
        push!(D_list, linkdims(mps))
        if print_details
            _, y_curr = mps_to_list_efficient(mps, sites, 500)
            println("Step $(step): max = ", maximum(real(y_curr)),
                    " min = ", minimum(real(y_curr)),
                    " linkdims = ", linkdims(mps))
        else
            println("Step $(step): ", linkdims(mps))
        end
    end
    return mps_list, D_list
end

mps_list, D_list = time_evolution(steps, mps_list, mpo_A, dt, derivative_mpo)

# Sanity check: how much did the solution change?
_, y_initial = mps_to_list_efficient(mps_list[1], sites, 500)
_, y_final   = mps_to_list_efficient(mps_list[end], sites, 500)
println("Max difference initial vs final: ", maximum(abs.(real(y_final) - real(y_initial))))

# Plot all snapshots with correct x-axis
p = Plots.plot(legend = true)
for (element_idx, element) in enumerate(mps_list[1:plot_every:end])
    t_current = (element_idx - 1) * plot_every * dt
    _, y = mps_to_list_efficient(element, sites, 500)
    Plots.plot!(p, x_plot, real(y), label = "t=$(round(t_current, digits=4))") #, xlims = (-2.0, 2.0))
end
display(p)

println("Finished.")