using ITensors
using LinearAlgebra
using Plots
using QuanticsTCI
using TCIITensorConversion
using FFTW

Nu, Nx, umin, umax = 6, 8, -0.1, 0.1
dx = (2*pi)/(Nx-1)
du = (umax - umin)/(2^Nu-1)

function decimal_to_padded_binary_list(decimal, bit_length)

    """

    Converts a decimal number to a binary of length bit_length

    Inputs:

    decimal = integer to be converted to binary

    bit_length = the length of the bit number i.e. the number of bits (this is used for padding zeros on the left hand side of the binary number)

    Output:

    binary_list = list of ones and zeros representing the decimal input in binary

    """


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

function tt_cross(tt_function, x_values, maxbonddim, tolerance)

    qtt, ranks, errors = quanticscrossinterpolate(Float64, tt_function, [x_values]; tolerance = tolerance, maxbonddim = maxbonddim)
        
    mps = MPS(qtt.tci)

    replace_list_sites = ["n=$(i)" => "S=1/2,Site,n=$(i)" for i in 1:Nu*Nx]
    replace_list_links = ["l=$(i),link" => "Link,l=$(i)" for i in 1:Nu*Nx-1]
    replacetags!(mps, replace_list_sites...)
    replacetags!(mps, replace_list_links...)

    return siteinds(mps), mps

end

function get_function_value(u)

    # Here we need to convert u from a linear index to a binary

    u_binary = decimal_to_padded_binary_list(u-1, Nx*Nu)

    # Here we need to first construct the actual u(x) vector which has length Nx

    u_x = Array{Float64}(undef, Nx)

    u_levels = [umin + i*du for i in 0:2^Nu-1]

    for i in 1:Nx

        bit = u_binary[(i-1)*Nu + 1 : (i-1)*Nu + Nu]
        binary_str = join([x == 1 ? "0" : "1" for x in bit], "")
        dec = parse(Int, binary_str; base = 2) + 1
        u_x[i] = u_levels[dec]

    end

    # Now we need u_x dot D acting on u_x as a number

    udu = dot(u_x, circshift(u_x, 1) .- 2 .* u_x .+ circshift(u_x, -1))/dx^2

    return exp(0.5*udu)

end

function mps_to_list(mps, N; reverse_flag = false)
    
    res = []
    tmp = contract(mps)
    for i in 1:2^N
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        if reverse_flag
            binary_list = reverse(decimal_to_padded_binary_list(i-1, N))
        else
            binary_list = decimal_to_padded_binary_list(i-1, N) 
        end
        push!(res, tmp[binary_list...])
    end
    return res

end

let

u_values = 1:2^(Nu*Nx)
maxbonddim = 100 
tolerance = 1e-5
sites, mps = tt_cross(get_function_value, u_values, maxbonddim, tolerance)

println(linkdims(mps))

mps_contracted = contract(mps[1:Nu])
for i in Nu+1:Nu*Nx
    idx = inds(mps[i]; :tags => "Site")[1]
    # mps_contracted *= mps[i]*onehot(idx => 1)
    mps_contracted *= mps[i]*delta(idx)
end
final_result = mps_to_list(mps_contracted, Nu)*du^(Nx-1)
plot(log.(abs.(final_result)), legend = false)
title!("log(abs(P(u(x=1)))) vs u(x=1)")

# for i in Nu+1:Nu*Nx
#     idx = inds(mps[i]; :tags => "Site")[1]
#     mps[i] = mps[i]*delta(idx)
# end

# tmp = contract(mps)
# res = []
# for i in 1:2^Nu
#     binary_list = decimal_to_padded_binary_list(i-1, Nu) 
#     push!(res, tmp[binary_list...])
# end

# res = []
# tmp = contract(mps)
# for element in u_values
#     u_binary = decimal_to_padded_binary_list(element-1, Nx*Nu)
#     push!(res, tmp[u_binary...])
# end

end
