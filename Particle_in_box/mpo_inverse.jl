using ITensors
using KrylovKit
using LinearAlgebra
import ITensors.svd as ITensors_SVD

function get_cost_fn_part_one(mpo, trial)

    N = length(mpo)
    res = prime(trial[1]; :tags => "Site")*mpo[1]*prime(dag(mpo[1]); :tags => "Link")*dag(trial[1]')
    for i in 2:N
        res *= prime(trial[i]; :tags => "Site")*mpo[i]*prime(dag(mpo[i]); :tags => "Link")*dag(trial[i]')
    end
    return res[1]

end

function get_cost_fn_part_two(mpo, trial)

    N = length(mpo)
    res = setprime(mpo[1]', 0; :plev => 2)*setprime(dag(trial[1]), 1; :tags => "Link")
    for i in 2:N
        res *= setprime(mpo[i]', 0; :plev => 2)*setprime(dag(trial[i]), 1; :tags => "Link")
    end
    return res[1]

end

function get_cost_fn(mpo, trial)

    N = length(mpo)
    return get_cost_fn_part_one(mpo, trial) - 2*get_cost_fn_part_two(mpo, trial) + 2^N

end

function get_inverse(mpo, sites, cutoff, max_sweeps, tol)

    
    # This function finds the inverse MPO of the mpo input
    
    # We are essentially solving M v = N_tilde where M and N_tilde are tensor networks and v is the solution to our problem for a given pair of sites and we optimize sweeping left and right the
    # trial solution until it reaches the maximum number of sweeps

    # SVDs below are affected by the cutoff input

    # What the function essentially does is SVD on M to get M = USV and then v is given by v = V^-1 S^-1 U^-1 N_tilde but we know U^-1 and V^-1 because U and V are unitary
    # hence U^-1, V^-1 = U^\dagger, V^\dagger

    # We initialize arrays left_right_parts_M, left_right_parts_N_tilde to help us compute M and N_tilde and we update them during the optimization

    # Tol is for the stopping condition on the cost function fractional change
    
    N = length(sites)
    trial = MPO(sites, "Id")
    ITensors.orthogonalize!(trial, 1)

    # Initialize left_right_parts_M 
    left_right_parts_M = []
    for i in 1:N
        # We are careful here and below wherever we are priming and unpriming indices to contract the right indices together 
        push!(left_right_parts_M, prime(trial[i]; :tags => "Site")*mpo[i]*prime(dag(mpo[i]); :tags => "Link")*dag(trial[i]'))
    end

    # Initialize left_right_parts_N_tilde 
    left_right_parts_N_tilde = []
    for i in 1:N
        push!(left_right_parts_N_tilde, setprime(mpo[i]', 0; :plev => 2)*setprime(dag(trial[i]), 1; :tags => "Link"))
    end

    E = 0.0
    
    # Start optimization
    for sw in 1:max_sweeps

        # Optimize by sweeping the trial MPO from left to right
        for i in 1:N-1
            # Compute M
            if N == 2
                M = setprime(setprime(prime(mpo[i]*mpo[i+1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i+1]'), 1; :plev => 2)
            else
                M = setprime(setprime(prime(mpo[i]*mpo[i+1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i+1]'), 1; :plev => 2)*prod(left_right_parts_M[setdiff(1:N, [i, i+1])])
            end
            
            # Compute the indices of M to be on U of SVD's M = USV
            bot = inds(M; :plev => 1) 
            
            # Computer N_tilde
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i+1]')*prod(left_right_parts_N_tilde[setdiff(1:end, [i, i+1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            # Do SVD on M
            U, S, V = ITensors.svd(M, bot..., cutoff = cutoff)

            # Get v named here v
            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))
            v = dag(V)*S_inv*dag(U)*N_tilde
            v = setprime(v, 1; :plev => 2)

            # SVD on v so as to update the two sites of the trial MPO
            U, S, V = ITensors.svd(v, commoninds(v, trial[i]), lefttags = "Link,l=$(i)", righttags = "Link,l=$(i)", cutoff = cutoff)
            trial[i], trial[i+1] = U, S*V
        
            # Update the left_right_parts_M
            for j in [i, i+1]                
                left_right_parts_M[j] = prime(trial[j]; :tags => "Site")*mpo[j]*prime(dag(mpo[j]); :tags => "Link")*dag(trial[j]')
            end

            # Update the left_right_parts_N_tilde
            for j in [i, i+1]
                left_right_parts_N_tilde[j] = setprime(mpo[j]', 0; :plev => 2)*setprime(dag(trial[j]), 1; :tags => "Link")
            end
        end

        # Same as above but now sweeping from right to left
        for i in N:-1:2

            if N == 2
                M = setprime(setprime(prime(mpo[i]*mpo[i-1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i-1]'), 1; :plev => 2)
            else
                M = setprime(setprime(prime(mpo[i]*mpo[i-1]; :tags => "Site"), 0; :plev => 2)*dag(mpo[i]')*dag(mpo[i-1]'), 1; :plev => 2)*prod(left_right_parts_M[setdiff(1:N, [i, i-1])])
            end
            
            bot = inds(M; :plev => 1) 
            
            if N == 2
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')
            else
                N_tilde = dag(mpo[i]')*dag(mpo[i-1]')*prod(left_right_parts_N_tilde[setdiff(1:end, [i, i-1])])
            end
            N_tilde = setprime(prime(N_tilde; :tags => "Site"), 1; :plev => 3)

            U, S, V = ITensors.svd(M, bot..., cutoff = cutoff)

            S_inv = ITensor(diagm(diag(Array(S, inds(S))).^(-1)), inds(S))
            v = dag(V)*S_inv*dag(U)*N_tilde
            v = setprime(v, 1; :plev => 2)

            U, S, V = ITensors.svd(v, commoninds(v, trial[i-1]), lefttags = "Link,l=$(i-1)", righttags = "Link,l=$(i-1)", cutoff = cutoff)
            
            trial[i-1], trial[i] = U*S, V

            for j in [i, i-1]                
                left_right_parts_M[j] = prime(trial[j]; :tags => "Site")*mpo[j]*prime(dag(mpo[j]); :tags => "Link")*dag(trial[j]')
            end

            for j in [i, i-1]                
                left_right_parts_N_tilde[j] = setprime(mpo[j]', 0; :plev => 2)*setprime(dag(trial[j]), 1; :tags => "Link")
            end
        end

        # Set the current energy variable E_curr to the final decreased energy E after a full sweep
        E = get_cost_fn(mpo, trial)
        
        # Check for stopping conditions
        if (E < tol)
            println("Energy accuracy reached for inverse MPO.")
            break
        elseif (max_sweeps == sw)
            println("Maximum sweeps reached before reaching desired energy accuracy for inverse MPO.")
        end
    end
    return trial
        
end
