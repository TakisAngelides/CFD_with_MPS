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

N = 18
L_start = -10.0
L_end = 10.0
N_phys = N
dx = (L_end - L_start) / (2^N_phys - 1)  
x_values = range(L_start, L_end, length = 2^N)
dt = dx               
steps = 10000
nu = 0.01
cutoff = 1e-15
max_sweeps = 3
cutoff_inverse = 1e-15
tol = 1e-15
maxbonddim = 100
tolerance = 1e-15
plot_every = 1000
fixed_boundary_conditions = false
pbc = !fixed_boundary_conditions
print_details = false

# Getting the initial state

# sites, mps = initial_random_mps(x_values, maxbonddim, tolerance) # uniform distribution [-1/2, 1/2] at every spatial point
# mps = initial_smooth_mps(N, pi, pi, 1, 1, 1)
# mps = randomMPS(sites) + constant_mps(N, 0)
# mps = sin_mps(N, pi, 1, sites)
# sites, mps = tt_cross(x -> sin(pi*x), x_values, maxbonddim, tolerance)
# sites, mps = tt_cross(x -> exp(-(x^2)/2), x_values, maxbonddim, tolerance)
function tmp_fun(x)
    # return exp(-((x-1)^2)/2) - exp(-((x+1)^2)/2)
    return 1.0 * exp(-(x^2) / 0.5)
end
sites, mps = tt_cross(tmp_fun, x_values, maxbonddim, tolerance)


# In case we want to keep fixed boundary conditions
if fixed_boundary_conditions
    fixed_boundary_conditions_mpo = get_boundary_conditions_MPO(sites)
end

# Fix the boundary conditions of the initial state if we have fixed boundary conditions
if fixed_boundary_conditions
    mps = apply(fixed_boundary_conditions_mpo, mps; cutoff = cutoff)
end

# println("Initial linkdims ", linkdims(mps))
# display(Plots.plot(real(mps_to_list(mps))))

# First we find the inverse operator A of B = 25*I - 12*dt*L where L = nu*u_xx (subscript = derivative w.r.t. x)

function get_inverse_of_B(sites, dt, dx, cutoff_inverse, max_sweeps, tol, nu, pbc)

    N = length(sites)

    res_opsum = OpSum()
    res_opsum += 25,"Id",1

    L_opsum = OpSum()
    L_opsum = sr1(N, pbc)
    L_opsum -= 2,"Id",1
    L_opsum += sl1(N, pbc)

    res_opsum -= nu*12*dt*L_opsum/(dx^2)

    mpo = MPO(res_opsum, sites)

    res = get_inverse(mpo, sites, cutoff_inverse, max_sweeps, tol)

    return res

end

mpo_A = get_inverse_of_B(sites, dt, dx, cutoff_inverse, max_sweeps, tol, nu, pbc)

# Define an mpo which is a*(shift right - shift left) which we can call mpo1

function get_derivative_mpo(N, sites, dx)

    opsum = OpSum()

    opsum = sr1(N)/(2*dx)
    opsum -= sl1(N)/(2*dx)

    return MPO(opsum, sites)

end

derivative_mpo = get_derivative_mpo(N, sites, dx)

# Define a function that will perform the non-linear part N

function get_N(mps_list, t_shift, derivative_mpo)

    return -apply(get_mpo_out_of_mps(mps_list[end-t_shift]), apply(derivative_mpo, mps_list[end-t_shift]; cutoff = cutoff); cutoff = cutoff)

end

# Now we simply do the time evolution

mps_list = [mps, mps, mps, mps]

function time_evolution(steps, mps_list, mpo_A, dt, derivative_mpo)

    mps = mps_list[end]
    D_list = [linkdims(mps), linkdims(mps), linkdims(mps), linkdims(mps)]
    for step in 1:steps

        mps = apply(mpo_A, 48*mps_list[end] - 36*mps_list[end-1] + 16*mps_list[end-2] - 3*mps_list[end-3] + 48*dt*get_N(mps_list, 0, derivative_mpo) - 72*dt*get_N(mps_list, 1, derivative_mpo) + 48*dt*get_N(mps_list, 2, derivative_mpo) - 12*dt*get_N(mps_list, 3, derivative_mpo); cutoff = cutoff)
        if fixed_boundary_conditions
            mps = apply(fixed_boundary_conditions_mpo, mps; cutoff = cutoff)
        end
        push!(mps_list, mps)
        push!(D_list, linkdims(mps))
        if print_details
            println("Step $(step): ", linkdims(mps), " ", minimum(real(mps_to_list(mps))), " ", maximum(real(mps_to_list(mps))))
        else
            println("Step $(step): ", linkdims(mps))
        end

    end

    return mps_list, D_list

end

function mps_to_list_efficient(mps, sites, num_samples; reverse_flag = false)

    x_inds = []
    y = []

    step = (2^length(mps)-1)/(num_samples-1)
    samples = (Int(round(i*step)) for i in 0:num_samples-1)

    for decimal in samples
    
        state = digits(decimal; base=2, pad=N).+1

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

mps_list, D_list = time_evolution(steps, mps_list, mpo_A, dt, derivative_mpo)
# mps_list = map(x -> real(mps_to_list_efficient(x, sites, 100)), mps_list[1:plot_every:end])

# mps_list_heatmap = hcat(mps_list...)'
# display(Plots.heatmap(mps_list_heatmap))

# f1 = h5open("$(@__DIR__)/velocity.h5", "w")
# f2 = h5open("$(@__DIR__)/bond_dimension.h5", "w")
p = Plots.plot(legend = false)
for (element_idx, element) in enumerate(mps_list[1:plot_every:end])
    # write(f1, "$(element_idx)", element)
    # write(f2, "$(element_idx)", D_list[element_idx])
    # Plots.plot!(p, element) # , label = "$(element_idx)"
    x, y = mps_to_list_efficient(element, sites, 1000)
    Plots.plot!(p, x.*dx .- (L_end - L_start)/2, real(y)) #, xlims = (-1.0, 1.0)) # , label = "$(element_idx)"
end
display(p)

println("Finished.")
