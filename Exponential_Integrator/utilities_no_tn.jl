function get_initial_state(inputs)
    
    initial_state_name = inputs["initial_state_name"]
    x_start = inputs["x_start"]
    x_end = inputs["x_end"]

    if initial_state_name == "cos_sin"

        x_values_u0 = inputs["x_values_u0"]

        return cos.(x_values_u0./16).*(1 .+ sin.(x_values_u0./16))
    
    end

end

function rhs(uhat, dt, kx, nonlinear_coeff, N)

    # Here can perform aliasing

    return uhat - nonlinear_coeff .* 0.5 .* kx .* 1im .* rfft(irfft(uhat, N).^2) .* dt

end

function time_step_euler(uhat, dt, kx, prop, nonlinear_coeff, N)

    return rhs(uhat, dt, kx, nonlinear_coeff, N).*prop

end

function time_step_heun(uhat, dt, kx, prop, nonlinear_coeff, N)

    uone = rhs(uhat, dt, kx, nonlinear_coeff, N).*prop
    return 0.5.*(uhat.*prop + rhs(uone, dt, kx, nonlinear_coeff, N))
    
end

function time_evolution_euler(uhat, starting_iteration, iterations, dt, kx, prop, scheme, results_file, inputs, path_to_backup, t_total_initial, N)

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
        uhat = time_step_euler(uhat, dt, kx, prop, nonlinear_coeff, N)
        time_for_iteration = time() - t
        println("Step = $(i), t = $(time_for_iteration)")
        flush(stdout)
    
        # Check if the uhat needs to be saved to file and if yes then save it
        if i ∈ steps_to_save
            if save_uhat 
                write(results_file, "$(i)", uhat)
            end
            if save_real_space_statevector
                write(results_file, "s_$(i)", irfft(uhat, N))
            end
            if save_D_t
                write(results_file, "t_$(i)", time_for_iteration)
            end
        end

        # Do backup
        if (i ∈ steps_to_backup) || ((time() - t_total_initial > day_to_backup_in_seconds) && (!backup_from_time_already_done))
            backup_from_time_already_done = true
            h5open(path_to_backup, "w") do f
                write(f, "starting_iteration", i+1)
                write(f, "uhat0", uhat)
            end
        end

    end

end

function time_evolution_heun(uhat, starting_iteration, iterations, dt, kx, prop, scheme, results_file, inputs, path_to_backup, t_total_initial, N)

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
        uhat = time_step_heun(uhat, dt, kx, prop, nonlinear_coeff, N)
        time_for_iteration = time() - t
        println("Step = $(i), t = $(time_for_iteration)")
        flush(stdout)
    
        # Check if the uhat needs to be saved to file and if yes then save it
        if i ∈ steps_to_save
            if save_uhat 
                write(results_file, "$(i)", uhat)
            end
            if save_real_space_statevector
                write(results_file, "s_$(i)", irfft(uhat, N))
            end
            if save_D_t
                write(results_file, "t_$(i)", time_for_iteration)
            end
        end

        # Do backup
        if (i ∈ steps_to_backup) || ((time() - t_total_initial > day_to_backup_in_seconds) && (!backup_from_time_already_done))
            backup_from_time_already_done = true
            h5open(path_to_backup, "w") do f
                write(f, "starting_iteration", i+1)
                write(f, "uhat0", uhat)
            end
        end

    end

end
