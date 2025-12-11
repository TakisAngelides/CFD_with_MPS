function sr1(N, pbc = true)

    opsum = OpSum()

    for n in 1:N
        opsum_tmp = OpSum()
        opsum_tmp += "S+",n
        for i in n+1:N
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S-",1
        for i in 2:N
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp
    end

    return opsum

end

function sl1(N, pbc = true)

    opsum = OpSum()

    for n in 1:N           
        opsum_tmp = OpSum()
        opsum_tmp += "S-",n
        for i in n+1:N
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp            
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S+",1
        for i in 2:N
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp
    end

    return opsum

end

function sr2(N, pbc = true)

    opsum = OpSum()

    for n in 1:N-1
        opsum_tmp = OpSum()
        opsum_tmp += "S-",n
        for i in n+1:N-1
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S+",1
        for i in 2:N-1
            opsum_tmp *= "S+",i
        end
        opsum += opsum_tmp
    end

    return opsum

end

function sl2(N, pbc = true)

    opsum = OpSum()

    for n in 1:N-1           
        opsum_tmp = OpSum()
        opsum_tmp += "S+",n
        for i in n+1:N-1
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp            
    end
    if pbc 
        opsum_tmp = OpSum()
        opsum_tmp += "S-",1
        for i in 2:N-1
            opsum_tmp *= "S-",i
        end
        opsum += opsum_tmp
    end

    return opsum

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

function mps_to_list(mps)
    
    N = length(mps)
    res = []
    tmp = contract(mps)
    for i in 0:2^N-1
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        binary_list = decimal_to_padded_binary_list(i, N)
        push!(res, tmp[binary_list...])
    end
    return res
end

function sin_mps(N, a, b, sites)

    # This gets the mps for b*sin(ax)

    mps = MPS(sites)
    links = [Index(2, "Link,l=$(n)") for n in 1:N-1]

    for n in 1:N

        if n == 1

            s, lr = sites[n], links[n]
            
            mps[n] = ITensor(ComplexF64, s, lr)

            mps[n][s => 1, lr => 1] = b/(2*1im)
            mps[n][s => 2, lr => 1] = b*exp(1im*a*2^(N-n)*dx)/(2*1im)
            
            mps[n][s => 1, lr => 2] = -b/(2*1im)
            mps[n][s => 2, lr => 2] = -b*exp(-1im*a*2^(N-n)*dx)/(2*1im)

        elseif n == N

            s, ll = sites[n], links[n-1]
            
            mps[n] = ITensor(ComplexF64, s, ll)

            mps[n][s => 1, ll => 1] = 1
            mps[n][s => 2, ll => 1] = exp(1im*a*2^(N-n)*dx)

            mps[n][s => 1, ll => 2] = 1
            mps[n][s => 2, ll => 2] = exp(-1im*a*2^(N-n)*dx)

        else

            s, ll, lr = sites[n], links[n-1], links[n]

            mps[n] = ITensor(ComplexF64, s, ll, lr)

            mps[n][s => 1, ll => 1, lr => 1] = 1
            mps[n][s => 2, ll => 1, lr => 1] = exp(1im*a*2^(N-n)*dx)

            mps[n][s => 1, ll => 2, lr => 2] = 1
            mps[n][s => 2, ll => 2, lr => 2] = exp(-1im*a*2^(N-n)*dx)

        end
    end

    return mps

end

function cos_mps(N, a, b, sites)

    # This gets the mps for b*cos(ax)

    mps = MPS(sites)
    links = [Index(2, "Link,l=$(n)") for n in 1:N-1]

    for n in 1:N

        if n == 1

            s, lr = sites[n], links[n]
            
            mps[n] = ITensor(ComplexF64, s, lr)

            mps[n][s => 1, lr => 1] = b/2
            mps[n][s => 2, lr => 1] = b*exp(1im*a*2^(N-n)*dx)/2
            
            mps[n][s => 1, lr => 2] = b/2
            mps[n][s => 2, lr => 2] = b*exp(-1im*a*2^(N-n)*dx)/2

        elseif n == N

            s, ll = sites[n], links[n-1]
            
            mps[n] = ITensor(ComplexF64, s, ll)

            mps[n][s => 1, ll => 1] = 1
            mps[n][s => 2, ll => 1] = exp(1im*a*2^(N-n)*dx)

            mps[n][s => 1, ll => 2] = 1
            mps[n][s => 2, ll => 2] = exp(-1im*a*2^(N-n)*dx)

        else

            s, ll, lr = sites[n], links[n-1], links[n]

            mps[n] = ITensor(ComplexF64, s, ll, lr)

            mps[n][s => 1, ll => 1, lr => 1] = 1
            mps[n][s => 2, ll => 1, lr => 1] = exp(1im*a*2^(N-n)*dx)

            mps[n][s => 1, ll => 2, lr => 2] = 1
            mps[n][s => 2, ll => 2, lr => 2] = exp(-1im*a*2^(N-n)*dx)

        end
    end

    return mps

end

function constant_mps(N, a, sites)

    # This gets the mps for the constant function of value a

    mps = MPS(sites)
    links = [Index(1, "Link,l=$(n)") for n in 1:N-1]

    for n in 1:N

        if n == 1

            s, lr = sites[n], links[n]
            
            mps[n] = ITensor(ComplexF64, s, lr)

            mps[n][s => 1, lr => 1] = a
            mps[n][s => 2, lr => 1] = a
            
        elseif n == N

            s, ll = sites[n], links[n-1]
            
            mps[n] = ITensor(ComplexF64, s, ll)

            mps[n][s => 1, ll => 1] = 1
            mps[n][s => 2, ll => 1] = 1

        else

            s, ll, lr = sites[n], links[n-1], links[n]

            mps[n] = ITensor(ComplexF64, s, ll, lr)

            mps[n][s => 1, ll => 1, lr => 1] = 1
            mps[n][s => 2, ll => 1, lr => 1] = 1

        end
    end

    return mps

end

function initial_smooth_mps(N, d, e, f, g, h, sites)

    # This gets the mps for u(x, t = 0) = h*cos(d*x)(f + g*sin(e*x))

    return apply(get_mpo_out_of_mps(cos_mps(N, d, h, sites)), (constant_mps(N, f, sites) + sin_mps(N, e, g, sites)))

end

function tt_cross(tt_function, x_values, maxbonddim, tolerance)

    qtt, ranks, errors = quanticscrossinterpolate(Float64, tt_function, [x_values]; tolerance = tolerance, maxbonddim = maxbonddim)
        
    mps = MPS(qtt.tci)

    replace_list_sites = ["n=$(i)" => "S=1/2,Site,n=$(i)" for i in 1:N]
    replace_list_links = ["l=$(i),link" => "Link,l=$(i)" for i in 1:N-1]
    replacetags!(mps, replace_list_sites...)
    replacetags!(mps, replace_list_links...)

    return siteinds(mps), mps

end

function initial_random_mps(x_values, maxbonddim, tolerance)

    function tt_function(x)
        return rand() - 0.5
    end
    mps = tt_cross(tt_function, x_values, maxbonddim, tolerance)

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