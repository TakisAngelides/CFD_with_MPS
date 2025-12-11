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

function get_prop(kappa, dt, kx2, delta_val, which_equation, inputs)

    kx2_coeff = inputs["kx2_coeff"]
    if which_equation == "B"
        return exp.(-kx2_coeff*dt.*kx2)
    else
        kx4_coeff = inputs["kx4_coeff"]
        return exp.(dt.*kx2.*(kx2_coeff .- kx4_coeff*kx2))
    end

end
