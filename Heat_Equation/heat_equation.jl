using ITensors
using Plots

function get_mpo(sites)

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

function get_initial_MPS(sites)
    # Prepare the N-dimensional tensor to be SVDd into an MPS
    tmp = ITensor(sites)
    for i in 1:2^N
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        binary_list = decimal_to_padded_binary_list(i-1, N) 
        tmp[binary_list...] = (1/pi^2)*sin(pi*(0 + (i-1)*dx))
    end
    # Perform SVD N-1 times to convert the N-dimensional tensor into an MPS of N sites
    mps = MPS(sites)
    for i in 1:N-1
        tmp_inds = inds(tmp)
        if i == 1
            U, S, V = svd(tmp, tmp_inds[1], lefttags = "Link,l=$(i)")
        else
            U, S, V = svd(tmp, tmp_inds[1:2]..., lefttags = "Link,l=$(i)", righttags = "Link,l=$(i)")
        end
        V = S*V
        tmp = V
        mps[i] = U
    end
    mps[N] = tmp
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

    final_mpo = add(id_mpo, -mpo; cutoff = 0)

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
        mps = apply(mpo, mps; cutoff = 1e-18)
        mps = apply(proj_mpo, mps; cutoff = 1e-18) # to set the boundaries to 0
        push!(res, mps)
        # tmp = mps[3]*mps[4]
        # u, s, v = svd(tmp, uniqueinds(mps[3], mps[4]))
        # println(diag(s), " ", linkdims(mps)[3])
        println(inds(mps[1]; :tags => "Link"))
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

    color_names = [:red, :green, :blue, :cyan, :magenta, :yellow, :black, :white, :orange, :purple, :pink, :brown, :gold, :silver, :gray]

    plot(legend = false)
    x = 0:dx:1

    for (t, mps) in enumerate(mps_list)
        
        if (t-1)%(div(n_steps,10))==0
        
            plot!(x, mps_to_list(mps), label = "MPS: t = $(round((t-1)*dt, digits = 1))", color = :black, alpha = 1/(t*0.005))
            scatter!(x, (1/pi^2)*exp(-(t-1)*dt)*sin.(pi*x), label = "Theory: t = $(round((t-1)*dt, digits = 1))", color = :black, alpha = 1/(t*0.005))
        
        end

    end

    title!("Velocity field vs position\nN = $(N), dt = $(dt)")
    savefig("Heat_Equation/evolution.png")

end

# Constants
N = 6
dx = 1/(2^N-1)
dt = 0.0001
# println(dx, " ", dt)
A = dt/(pi^2*dx^2)
B = 1-(2*dt)/(pi^2*dx^2)
n_steps = 100

# Initialize the site indices
sites = siteinds("S=1/2", N)

# Get the time evolution MPO
mpo = get_mpo(sites)

# Get the projector MPO to take care of boundary conditions
proj_mpo = get_boundary_conditions_MPO(sites)

# Get the initial MPS according to the initial condition U(t = 0, x) = sin(pi * x)/pi^2
mps = get_initial_MPS(sites)

# # Perform time evolution according to the heat equation
mps_list = time_evolve(n_steps, mpo, mps, proj_mpo)

# # # Plot the results
plot_results(mps_list)

println("Finished.")
