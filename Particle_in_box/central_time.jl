using ITensors
using Plots
include("mpo_inverse.jl")

function get_mpo(sites, A, B)

    # Define links indices and initialize MPO of time evolution
    links = [Index(3, "Link,l=$n") for n in 1:N-1]
    mpo = MPO(sites)

    # First tensor of MPO
    s1, s2, l1 = prime(sites[1]), dag(sites[1]), links[1]
    mpo[1] = ITensor(l1, s1, s2)
    # Id
    mpo[1][l1 => 1, s1 => 1, s2 => 1] = 1 
    mpo[1][l1 => 1, s1 => 2, s2 => 2] = 1
    # sigma_01 = [[0 1], [0 0]] = Up
    mpo[1][l1 => 2, s1 => 1, s2 => 2] = 1
    # sigma_10 = [[0 0], [1 0]] = Down
    mpo[1][l1 => 3, s1 => 2, s2 => 1] = 1

    # Set the bulk MPO 
    for i in 2:N-1

        s1, s2, l1, l2 = prime(sites[i]), dag(sites[i]), links[i-1], links[i]
        mpo[i] = ITensor(l1, l2, s1, s2)
        # Id
        mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
        mpo[i][l1 => 1, l2 => 1, s1 => 2, s2 => 2] = 1
        # Up
        mpo[i][l1 => 1, l2 => 2, s1 => 1, s2 => 2] = 1
        # Down 
        mpo[i][l1 => 1, l2 => 3, s1 => 2, s2 => 1] = 1
        # Up
        mpo[i][l1 => 3, l2 => 3, s1 => 1, s2 => 2] = 1
        # Down
        mpo[i][l1 => 2, l2 => 2, s1 => 2, s2 => 1] = 1

    end

    # Last tensor of MPO
    s1, s2, l1 = prime(sites[N]), dag(sites[N]), links[N-1]
    mpo[N] = ITensor(l1, s1, s2)
    # Id
    mpo[N][l1 => 1, s1 => 1, s2 => 1] = B
    mpo[N][l1 => 1, s1 => 2, s2 => 2] = B
    # Up
    mpo[N][l1 => 1, s1 => 1, s2 => 2] = A
    # Down
    mpo[N][l1 => 1, s1 => 2, s2 => 1] = A
    # Down
    mpo[N][l1 => 2, s1 => 2, s2 => 1] = A
    # Up
    mpo[N][l1 => 3, s1 => 1, s2 => 2] = A

    return mpo

end

function decimal_to_padded_binary_list(decimal, bit_length)
    binary_list = Int[]

    while decimal > 0 || length(binary_list) < bit_length
        pushfirst!(binary_list, (decimal % 2) + 1)
        decimal = div(decimal, 2)
    end

    # Pad with leading zeros if needed
    while length(binary_list) < bit_length
        pushfirst!(binary_list, 0)
    end

    return binary_list 
end

function get_initial_MPS_by_SVD(sites)
    
    # Prepare the N-dimensional tensor to be SVDd into an MPS
    tmp = ITensor(sites)
    for (i_idx, i) in enumerate(1:2^N)
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        binary_list = decimal_to_padded_binary_list(i-1, N) 
        tmp[binary_list...] = (1/pi^2)*sin(pi*(0 + (i-1)*dx))
        # if i_idx == 2^(N-1)
        #     tmp[binary_list...] = 1
        # else
        #     tmp[binary_list...] = 0
        # end
    end
    # Perform SVD N-1 times to convert the N-dimensional tensor into an MPS of N sites
    mps = MPS(sites)
    for i in 1:N-1
        tmp_inds = inds(tmp)
        if i == 1
            U, S, V = svd(tmp, tmp_inds[1], lefttags = "Link,l=$(i)", cutoff = 1e-15)
        else
            U, S, V = svd(tmp, tmp_inds[1:2]..., lefttags = "Link,l=$(i)", righttags = "Link,l=$(i)", cutoff = 1e-15)
        end
        println(S)
        V = S*V
        tmp = V
        mps[i] = U
    end
    mps[N] = tmp
    return mps

end

function get_initial_MPS(sites, alpha)

    """
    This function gets specifically the MPS = alpha * sin(pi * x)
    """
    
    mps = MPS(sites)
    links = [Index(2, "Link,l=$(n)") for n in 1:N-1]

    for n in 1:N

        if n == 1

            s, lr = sites[n], links[n]
            
            mps[n] = ITensor(ComplexF64, s, lr)

            mps[n][s => 1, lr => 1] = alpha/(2*1im)
            mps[n][s => 2, lr => 1] = alpha*exp(1im*pi*2^(N-n)*dx)/(2*1im)
            
            mps[n][s => 1, lr => 2] = -alpha/(2*1im)
            mps[n][s => 2, lr => 2] = -alpha*exp(-1im*pi*2^(N-n)*dx)/(2*1im)

        elseif n == N

            s, ll = sites[n], links[n-1]
            
            mps[n] = ITensor(ComplexF64, s, ll)

            mps[n][s => 1, ll => 1] = 1
            mps[n][s => 2, ll => 1] = exp(1im*pi*2^(N-n)*dx)

            mps[n][s => 1, ll => 2] = 1
            mps[n][s => 2, ll => 2] = exp(-1im*pi*2^(N-n)*dx)

        else

            s, ll, lr = sites[n], links[n-1], links[n]

            mps[n] = ITensor(ComplexF64, s, ll, lr)

            mps[n][s => 1, ll => 1, lr => 1] = 1
            mps[n][s => 2, ll => 1, lr => 1] = exp(1im*pi*2^(N-n)*dx)

            mps[n][s => 1, ll => 2, lr => 2] = 1
            mps[n][s => 2, ll => 2, lr => 2] = exp(-1im*pi*2^(N-n)*dx)

        end
    end

    return mps

end

function get_boundary_conditions_MPO(sites)

    # The bond dimension 2 matrices of the MPS for the GHZ state |0...0> + |1...1> are A_sigma=1 = [[1 0], [0 0]], A_sigma=2 = [[0 0], [0 1]]
    # Our boundary conditions MPO needs to be Id - |0...0><0...0| - |1...1><1...1| so we will put a delta function_sigma,sigma' on the GHZ MPS
    # to convert it to an MPO

    links = [Index(2, "Link,l=$n") for n in 1:N-1]
    mpo = MPO(sites) # This will store the MPO |0...0><0...0| + |1...1><1...1|

    for (site_idx, site) in enumerate(sites)

        if site_idx == 1

            s1, s2, l1 = prime(site), dag(site), links[1]
            mpo[1] = ITensor(l1, s1, s2)
            # A_sigma=1 = [1 0]
            mpo[site_idx][l1 => 1, s1 => 1, s2 => 1] = 1
            # A_sigma=2 = [0 1]
            mpo[site_idx][l1 => 2, s1 => 2, s2 => 2] = 1

        elseif site_idx == N

            s1, s2, l1 = prime(site), dag(site), links[N-1]
            mpo[N] = ITensor(l1, s1, s2)
            # A_sigma=1 = [1 0]
            mpo[site_idx][l1 => 1, s1 => 1, s2 => 1] = 1
            # A_sigma=2 = [0 1]
            mpo[site_idx][l1 => 2, s1 => 2, s2 => 2] = 1

        else

            s1, s2, l1, l2 = prime(site), dag(site), links[site_idx-1], links[site_idx]
            mpo[site_idx] = ITensor(l1, l2, s1, s2)
            # A_sigma=1 = [[1 0], [0 0]]
            mpo[site_idx][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
            # A_sigma=2 = [[0 0], [0 1]]
            mpo[site_idx][l1 => 2, l2 => 2, s1 => 2, s2 => 2] = 1

        end

    end

    id_mpo = MPO(sites, "Id")

    final_mpo = id_mpo - mpo

    return final_mpo

end

function time_evolve(n_steps, mpo, mps, proj_mpo)

    """
    n_steps = how many steps to time evolve

    mpo = the differential equation time evolver mpo

    mps = the initial mps to time evolve

    proj_mpo = a projector mpo for the boundary conditions to be maintained

    """

    res = [mps]
    for _ in 1:n_steps
        mps = apply(mpo, mps)
        mps = apply(proj_mpo, mps) # to set the boundaries to 0
        push!(res, mps)
    end
    return res
end

function mps_to_list(mps)
    
    res = []
    tmp = contract(mps)
    for i in 1:2^N
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        binary_list = decimal_to_padded_binary_list(i-1, N) 
        push!(res, tmp[binary_list...])
    end
    return res

end

function plot_results(mps_list)
    
    plot(legend = false)
    x = 0:dx:1
    for (t, mps) in enumerate(mps_list)
        if (t-1)%(div(n_steps, 5))==0
            plot!(x, real(mps_to_list(mps)), label = "MPS: t = $(round((t-1)*dt, digits = 1))", color = :black, alpha = 1/(t*0.005))
            scatter!(x, (1/pi^2)*exp(-(t-1)*dt)*sin.(pi*x), label = "Theory: t = $(round((t-1)*dt, digits = 1))", color = :black, alpha = 1/(t*0.005))
        end
    end
    title!("Velocity field vs position\nN = $(N), dt = $(dt)")
    savefig("Heat_Equation/evolution_new.png")

end

function make_animation()
    
    x = 0:dx:1
    
    # Set up the Plots library for animation
    gr()

    # Function to create the frame for the animation
    function create_frame(t, mps)

        plot(x, real(mps_to_list(mps)), xlims=(0, 1), ylims=(0, alpha))
        # plot!(x, (1/pi^2)*exp(-(t-1)*dt)*sin.(pi*x), linestyle = :dash)
        title!("t = $t")

    end

    # Number of frames in the animation
    num_frames = 50

    # Create and save the animation
    animation = @animate for (t, mps) in enumerate(mps_list)
        create_frame(t, mps)
    end

    gif(animation, "animation.gif", fps = num_frames)

end

# Constants
N = 4
dx = 1/(2^N-1)
dt = (1e-2)
# println(dx, " ", dt)
alpha = -1
a = dt*alpha/(2*dx^2)
b = 1-(dt*alpha/(dx^2))
c = -dt*alpha/(2*dx^2)
d = 1+(dt*alpha/(dx^2))
n_steps = 500
plot_every = 1
max_sweeps = 50
cutoff = 1e-12
tol = 1e-16

# Initialize the site indices
sites = siteinds("S=1/2", N)

# Get the time evolution MPO
mpo_r = get_mpo(sites, a, b)
mpo_l = get_mpo(sites, c, d)
mpo_l_inv = get_inverse(mpo_l, sites, cutoff, max_sweeps, tol)
# println(tr(apply(mpo_l_inv, mpo_l) - MPO(sites,"Id")))
mpo = apply(mpo_l_inv, mpo_r)

# Get the projector MPO to take care of boundary conditions
proj_mpo = get_boundary_conditions_MPO(sites)

# Get the initial MPS according to the initial condition U(t = 0, x) = sin(pi * x)/pi^2
mps = get_initial_MPS(sites, -alpha)

# Perform time evolution according to the heat equation
mps_list = time_evolve(n_steps, mpo, mps, proj_mpo)

# Plot the results
plot_results(mps_list)

# # Make the animation
# make_animation()
