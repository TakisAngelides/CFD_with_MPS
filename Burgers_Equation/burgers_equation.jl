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

N = 14
sites = siteinds("S=1/2", N)
dt = 0.001
dx = 0.001
cutoff = 1e-15
steps = 10000
nu = 0.01
# for mpo inverse
cutoff_inverse = 1e-13
max_sweeps = 5
tol = 1e-11
# for TT cross
x_values = range(-7, 7, length = 2^N)
maxbonddim = 100 
tolerance = 1e-12
# for random initial state reproducibility of results
Random.seed!(1234)
# for tracking the minimum and maximum
print_details = false
# for the boundary conditions 
fixed_boundary_conditions = false
pbc = !fixed_boundary_conditions
# how many time steps to plot
plot_every = 1000

# Getting the initial state

# sites, mps = initial_random_mps(x_values, maxbonddim, tolerance) # uniform distribution [-1/2, 1/2] at every spatial point
# mps = initial_smooth_mps(N, pi, pi, 1, 1, 1)
# mps = randomMPS(sites) + constant_mps(N, 0)
# mps = sin_mps(N, pi, 1, sites)
# sites, mps = tt_cross(x -> sin(pi*x), x_values, maxbonddim, tolerance)
# sites, mps = tt_cross(x -> exp(-(x^2)/2), x_values, maxbonddim, tolerance)
function tmp_fun(x)
    return exp(-((x-1)^2)/2) - exp(-((x+1)^2)/2)
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
    L_opsum = sr1(N, pbc)/(dx^2)
    L_opsum -= 2/(dx^2),"Id",1
    L_opsum += sl1(N, pbc)/(dx^2)

    res_opsum -= nu*12*dt*L_opsum

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

mps_list, D_list = time_evolution(steps, mps_list, mpo_A, dt, derivative_mpo)
mps_list = map(x -> real(mps_to_list(x)), mps_list[1:plot_every:end])

# mps_list_heatmap = hcat(mps_list...)'
# display(Plots.heatmap(mps_list_heatmap))

f1 = h5open("/Users/takisangelides/CFD_MPS/Burgers_Equation/velocity.h5", "w")
f2 = h5open("/Users/takisangelides/CFD_MPS/Burgers_Equation/bond_dimension.h5", "w")
p = Plots.plot(legend = false)
for (element_idx, element) in enumerate(mps_list)
    write(f1, "$(element_idx)", element)
    write(f2, "$(element_idx)", D_list[element_idx])
    Plots.plot!(p, element) # , label = "$(element_idx)"
end
display(p)

println("Finished.")
