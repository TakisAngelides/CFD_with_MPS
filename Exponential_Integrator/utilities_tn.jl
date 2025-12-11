function get_hadamard_mpo(sites, idx)

    opsum = OpSum()
    opsum += h,idx
    return MPO(opsum, sites)

end

function tt_cross(tt_function, x_values, maxbonddim, tolerance)

    qtt, ranks, errors = quanticscrossinterpolate(Float64, tt_function, [x_values]; tolerance = tolerance, maxbonddim = maxbonddim)
        
    mps = MPS(qtt.tci)

    n = length(mps)

    replace_list_sites = ["n=$(i)" => "S=1/2,Site,n=$(i)" for i in 1:n]
    replace_list_links = ["l=$(i),link" => "Link,l=$(i)" for i in 1:n-1]
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

    n_qubits = length(sites)
    mpo = get_hadamard_mpo(sites, 1)
    for i in 1:n_qubits-1
        mpo = apply(get_phase_mpo(sites, i, qft_sign_convention), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i + 1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

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

function reverse_mps(mps, sites)
    mps_reversed = MPS(sites)
    N=length(sites)
    for i in 1:N
        mps_reversed[i] = mps[N-i+1]
    end
    return mps_reversed
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

function convert_evolution_to_real_space(uhats, qft_inv_mpo, cutoff)

    us = []

    for uhat in uhats

        u_hat_reverse = reverse_mps(uhat, sites)

        push!(us, apply(qft_inv_mpo, u_hat_reverse; cutoff = cutoff))

    end

    return us

end

function convert_state_to_real_space(uhat, qft_inv_mpo, cutoff, sites)

    u_hat_reverse = reverse_mps(uhat, sites)

    return apply(qft_inv_mpo, u_hat_reverse; cutoff = cutoff)

end

function sin_mps(a, b, sites, x_start, x_end)

    # This gets the mps for b*sin(ax)

    mps = MPS(sites)
    N = length(sites)
    dx = (x_end - x_start) / 2^N
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

function get_initial_state(inputs)
    
    initial_state_name = inputs["initial_state_name"]
    x_start = inputs["x_start"]
    x_end = inputs["x_end"]

    if initial_state_name == "sin_tt"

        max_bond_ttc_u0 = inputs["max_bond_ttc_u0"]
        tol_ttc_u0 = inputs["tol_ttc_u0"]
        x_values_u0 = inputs["x_values_u0"]

        sites, u0 = tt_cross(x -> sin(x*2*pi/(x_end-x_start)), x_values_u0, max_bond_ttc_u0, tol_ttc_u0) # sin(x*2*pi/(x_end-x_start))

        return sites, u0

    elseif initial_state_name == "-sin"

        n_qubits = inputs["n_qubits"]
        sites = siteinds("S=1/2", n_qubits)
        u0 = sin_mps(2*pi/(x_end-x_start), -1, sites, x_start, x_end) # -sin(x*2*pi/(x_end-x_start))

        return sites, u0

    elseif initial_state_name == "sin"

        n_qubits = inputs["n_qubits"]
        sites = siteinds("S=1/2", n_qubits)
        u0 = sin_mps(2*pi/(x_end-x_start), 1, sites, x_start, x_end) # sin(x*2*pi/(x_end-x_start))

        return sites, u0

    elseif initial_state_name == "exp_tt"

        max_bond_ttc_u0 = inputs["max_bond_ttc_u0"]
        tol_ttc_u0 = inputs["tol_ttc_u0"]
        x_values_u0 = inputs["x_values_u0"]

        sites, u0 = tt_cross(x -> -exp(-x^2), x_values_u0, max_bond_ttc_u0, tol_ttc_u0) # -exp(-x^2)

        return sites, u0

    elseif initial_state_name == "analytical_sol_tt"

        max_bond_ttc_u0 = inputs["max_bond_ttc_u0"]
        tol_ttc_u0 = inputs["tol_ttc_u0"]
        x_values_u0 = inputs["x_values_u0"]

        function analytical_sol_tt(x)
            return 1 - 45*tan(0.5*x) - 15*(tan(0.5*x))^3
        end

        sites, u0 = tt_cross(analytical_sol_tt, x_values_u0, max_bond_ttc_u0, tol_ttc_u0) # analytical solution from Feng 2012

        return sites, u0

    elseif initial_state_name == "cos_sin"

        max_bond_ttc_u0 = inputs["max_bond_ttc_u0"]
        tol_ttc_u0 = inputs["tol_ttc_u0"]
        x_values_u0 = inputs["x_values_u0"]

        function cos_sin_tt(x)
            return cos(x/16)*(1 + sin(x/16))
        end

        sites, u0 = tt_cross(cos_sin_tt, x_values_u0, max_bond_ttc_u0, tol_ttc_u0)

        return sites, u0
    
    end

end

function rhs(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, nonlinear_coeff)

    # Here can perform aliasing

    # Inverse fft of uhat:
    uhat_reversed = reverse_mps(uhat, sites)
    u = apply(qft_inv_mpo, uhat_reversed; cutoff = cutoff)

    # Squaring u
    u_as_mpo = get_mpo_out_of_mps(u)
    u_square = apply(u_as_mpo, u; cutoff = cutoff)

    #U square on frequency space
    u_square_hat = apply(qft_mpo, u_square; cutoff = cutoff) 

    return uhat - nonlinear_coeff*0.5*dt*1im*reverse_mps(apply(kx_mpo, u_square_hat; cutoff = cutoff), sites)

end

function euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)

    tmp1 = reverse_mps(rhs(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, nonlinear_coeff), sites)
    
    tmp2 = apply(prop_mpo, tmp1; cutoff = cutoff)
    
    return reverse_mps(tmp2, sites)

end

function heun(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)

    tmp1 = reverse_mps(rhs(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, nonlinear_coeff), sites)
    uhat_one = reverse_mps(apply(prop_mpo, tmp1 ; cutoff = cutoff) , sites)

    uhat_reverse = reverse_mps(uhat, sites)
    prop_uhat= reverse_mps(apply(prop_mpo, uhat_reverse; cutoff = cutoff) , sites)
    rhs_one= rhs(uhat_one, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, nonlinear_coeff)

    return 0.5*(prop_uhat + rhs_one)
    
end

function time_step_euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)

    return euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)
    
end 

function time_step_heun(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)

    return heun(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)

end 

function time_evolution_euler(uhat, starting_iteration, iterations, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, scheme, results_file, inputs, path_to_backup, t_total_initial)

    # Get list defining at which steps to make a back up and save
    save_all_steps = parse(Bool, inputs["save_all_steps"])
    if !save_all_steps
        steps_to_save = inputs["steps_to_save"]
    else
        steps_to_save = 1:iterations
    end
    steps_to_backup = inputs["steps_to_backup"]
    day_to_backup = inputs["day_to_backup"]
    day_to_backup_in_seconds = day_to_backup*86400
    backup_from_time_already_done = false
    save_real_space_statevector = parse(Bool, inputs["save_real_space_statevector"])
    save_uhat = parse(Bool, inputs["save_uhat"])
    save_D_t = parse(Bool, inputs["save_D_t"])
    nonlinear_coeff = inputs["nonlinear_coeff"]

    # Start the whole time evolution
    for i in starting_iteration:iterations

        # Do one time step evolution
        t = time()
        uhat = time_step_euler(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)
        time_for_iteration = time() - t
        println("Step = $(i), t = $(time_for_iteration), l = $(linkdims(uhat))")
        flush(stdout)
    
        # Check if the uhat needs to be saved to file and if yes then save it
        if i ∈ steps_to_save
            if save_uhat 
                write(results_file, "$(i)", uhat)
            end
            if save_real_space_statevector
                write(results_file, "s_$(i)", mps_to_list(convert_state_to_real_space(uhat, qft_inv_mpo, cutoff, sites)))
            end
            if save_D_t
                write(results_file, "D_$(i)", linkdims(uhat))
                write(results_file, "t_$(i)", time_for_iteration)
            end
        end

        # Do backup
        if (i ∈ steps_to_backup) || ((time() - t_total_initial > day_to_backup_in_seconds) && (!backup_from_time_already_done))
            backup_from_time_already_done = true
            h5open(path_to_backup, "w") do f
                write(f, "starting_iteration", i+1)
                write(f, "uhat0", uhat)
                write(f, "qft_mpo", qft_mpo)
                write(f, "qft_inv_mpo", qft_inv_mpo)
                write(f, "kx_mpo", kx_mpo)
                write(f, "prop_mpo", prop_mpo)
            end
        end

    end

end

function time_evolution_heun(uhat, starting_iteration, iterations, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, scheme, results_file, inputs, path_to_backup, t_total_initial)

    # Get list defining at which steps to make a back up and save
    save_all_steps = parse(Bool, inputs["save_all_steps"])
    if !save_all_steps
        steps_to_save = inputs["steps_to_save"]
    else
        steps_to_save = 1:iterations
    end
    steps_to_backup = inputs["steps_to_backup"]
    day_to_backup = inputs["day_to_backup"]
    day_to_backup_in_seconds = day_to_backup*86400
    backup_from_time_already_done = false
    save_real_space_statevector = parse(Bool, inputs["save_real_space_statevector"])
    save_uhat = parse(Bool, inputs["save_uhat"])
    save_D_t = parse(Bool, inputs["save_D_t"])
    nonlinear_coeff = inputs["nonlinear_coeff"]

    # Start the whole time evolution
    for i in starting_iteration:iterations

        # Do one time step evolution
        t = time()
        uhat = time_step_heun(uhat, sites, qft_inv_mpo, cutoff, qft_mpo, dt, kx_mpo, prop_mpo, nonlinear_coeff)
        time_for_iteration = time() - t
        println("Step = $(i), t = $(time_for_iteration), l = $(linkdims(uhat))")
        flush(stdout)
    
        # Check if the uhat needs to be saved to file and if yes then save it
        if i ∈ steps_to_save
            if save_uhat 
                write(results_file, "$(i)", uhat)
            end
            if save_real_space_statevector
                write(results_file, "s_$(i)", mps_to_list(convert_state_to_real_space(uhat, qft_inv_mpo, cutoff, sites)))
            end
            if save_D_t
                write(results_file, "D_$(i)", linkdims(uhat))
                write(results_file, "t_$(i)", time_for_iteration)
            end
        end

        # Do backup
        if (i ∈ steps_to_backup) || ((time() - t_total_initial > day_to_backup_in_seconds) && (!backup_from_time_already_done))
            backup_from_time_already_done = true
            h5open(path_to_backup, "w") do f
                write(f, "starting_iteration", i+1)
                write(f, "uhat0", uhat)
                write(f, "qft_mpo", qft_mpo)
                write(f, "qft_inv_mpo", qft_inv_mpo)
                write(f, "kx_mpo", kx_mpo)
                write(f, "prop_mpo", prop_mpo)
            end
        end

    end

end
