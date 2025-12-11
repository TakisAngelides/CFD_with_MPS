function get_hadamard_mpo(sites, idx)

    opsum = OpSum()
    opsum += h,idx
    return MPO(opsum, sites)

end

function tt_cross(tt_function, x_values, maxbonddim, tolerance)

    qtt, ranks, errors = quanticscrossinterpolate(Float64, tt_function, [x_values]; tolerance = tolerance, maxbonddim = maxbonddim, sweepstrategy = :backward)
        
    mps = MPS(qtt.tci)

    replace_list_sites = ["n=$(i)" => "S=1/2,Site,n=$(i)" for i in 1:N]
    replace_list_links = ["l=$(i),link" => "Link,l=$(i)" for i in 1:N-1]
    replacetags!(mps, replace_list_sites...)
    replacetags!(mps, replace_list_links...)

    return siteinds(mps), mps

end

function get_phase_mpo(sites, starting_site, sign)

    # Define links indices and initialize MPO of time evolution
    N = length(sites)
    links = [n < starting_site ? Index(1, "Link,l=$n") : Index(2, "Link,l=$n") for n in 1:N-1]
    mpo = MPO(sites)

    # First tensor of MPO
    s1, s2, l1 = prime(sites[1]), dag(sites[1]), links[1]
    mpo[1] = ITensor(ComplexF64, l1, s1, s2)
    if starting_site == 1
        # |0><0|
        mpo[1][l1 => 1, s1 => 1, s2 => 1] = 1
        # |1><1|
        mpo[1][l1 => 2, s1 => 2, s2 => 2] = 1
    else
        # Id
        mpo[1][l1 => 1, s1 => 1, s2 => 1] = 1
        mpo[1][l1 => 1, s1 => 2, s2 => 2] = 1
    end 
    
    # Set the bulk MPO 
    for i in 2:N-1
        
        s1, s2, l1, l2 = prime(sites[i]), dag(sites[i]), links[i-1], links[i]
        mpo[i] = ITensor(ComplexF64, l1, l2, s1, s2)

        if i < starting_site
            # Id
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 1, l2 => 1, s1 => 2, s2 => 2] = 1   
        elseif i == starting_site
            # |0><0|
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1 
            # |1><1|
            mpo[i][l1 => 1, l2 => 2, s1 => 2, s2 => 2] = 1
        else
            # Id
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 1, l2 => 1, s1 => 2, s2 => 2] = 1
            # Phase matrix
            mpo[i][l1 => 2, l2 => 2, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 2, l2 => 2, s1 => 2, s2 => 2] = exp(sign*1im * pi / 2^(i - starting_site))
        end

    end

    # Last tensor of MPO
    s1, s2, l1 = prime(sites[N]), dag(sites[N]), links[N-1]
    mpo[N] = ITensor(ComplexF64, l1, s1, s2)
    # Id
    mpo[N][l1 => 1, s1 => 1, s2 => 1] = 1
    mpo[N][l1 => 1, s1 => 2, s2 => 2] = 1
    # Phase matrix
    mpo[N][l1 => 2, s1 => 1, s2 => 1] = 1
    mpo[N][l1 => 2, s1 => 2, s2 => 2] = exp(sign*1im * pi / 2^(N - starting_site))
    
    return mpo

end

function get_qft_mpo(sites, qft_sign_convention, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, 1)
    for i in 1:N-1
        mpo = apply(get_phase_mpo(sites, i, qft_sign_convention), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i + 1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

end

function reversed_mps(mps,sites)
    mps_reversed = MPS(sites)
    N=length(sites)
    for i in 1:N
        mps_reversed[i] = mps[N-i+1]
    end
    return mps_reversed
end

function get_inv_qft_mpo(sites, qft_sign_convention, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, N)
    for i in N-1:-1:1
        mpo = apply(get_phase_mpo(sites, i, -qft_sign_convention), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
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

function mps_to_list(mps; reverse_flag = false)::Vector{promote_itensor_eltype(mps)}
    
    res = []
    N = length(mps)
    tmp = contract(mps)
    for i in 1:2^(N)
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

function get_mpo_out_of_mps(mps)

    mpo = MPO(length(mps))

    for (idx, A) in enumerate(mps)

        s = inds(A; :tags => "Site")[1]
        d = delta(s, s', s'')
        mpo[idx] = A*d

    end

    setprime!(mpo, 0; :plev => 2)

    return mpo

end

function mpo_to_matrix(mpo)

    N = length(mpo)
    res = contract(mpo)
    res = Array(res, inds(res; :plev => 1)..., inds(res; :plev => 0)...)
    res = reshape(res, 2^N, 2^N)

    return res

end

function prop(kappa, dt, kx2, delta)

    return exp.(kappa*dt.*kx2.*(1 .- kx2).*delta)

end

function rhs(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo)

    # Here can perform aliasing

    # Inverse fft of uhat:
    uhat_reversed = reversed_mps(uhat, sites)
    u = apply(qft_inv_mpo, uhat_reversed; cutoff = cutoff)

    # Squaring u
    u_as_mpo = get_mpo_out_of_mps(u)
    u_square = apply(u_as_mpo, u; cutoff = cutoff)

    #U square on frequency space
    u_square_hat = apply(qft_mpo, u_square; cutoff = cutoff) 

    return uhat - 0.5*dt*1im*reversed_mps(apply(kx_mpo, u_square_hat; cutoff = cutoff), sites)

end

function euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo)

    tmp1 = reversed_mps(rhs(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo), sites)
    
    tmp2 = apply(prop_mpo, tmp1; cutoff = cutoff)
    
    return reversed_mps(tmp2, sites)

end

function time_step(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo)

    return euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo)

end 

function time_evolution(uhat, iters, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo)

    uhats = [uhat]

    for i in 1:iters

        println(i)

        uhat = time_step(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo)
        if i % 500 == 0
            u_hat_reverse = reversed_mps(uhat, sites)
            u = apply(qft_inv_mpo, u_hat_reverse; cutoff = cutoff)
            u = (u + conj(u))/2
            uhat = reversed_mps(apply(qft_mpo, u; cutoff = cutoff), sites)
        end
        push!(uhats, uhat)

    end

    return uhats

end

function convert_evolution_to_real_space(uhats, qft_inv_mpo, cutoff)

    us = []

    for uhat in uhats

        u_hat_reverse = reversed_mps(uhat, sites)

        push!(us, apply(qft_inv_mpo, u_hat_reverse; cutoff = cutoff))

    end

    return us

end
