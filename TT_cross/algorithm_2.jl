using LinearAlgebra
using ITensors
using Plots
using Random
using FileIO, Images, ImageTransformations
Random.seed!(1234)

# Implements algorithm 2 of https://www.sciencedirect.com/science/article/pii/S0010465519302565

function compute_approx_matrix(M, pivots)

    return (M[:, pivots[:, 2]]/M[pivots[:, 1], pivots[:, 2]])*M[pivots[:, 1], :]

end

function compute_total_error(M, pivots)

    return norm(M - compute_approx_matrix(M, pivots))

end

function compute_rand_pivots_error(M, rand_pivots, pivots)

    # Below we use diag to compute the error list of only the rand_pivot elements and not include all their combinations of rows and columns just the pairs which for the list rand_pivots
    return abs.(diag(M[rand_pivots[:, 1], rand_pivots[:, 2]]) - diag(((M[rand_pivots[:, 1], pivots[:, 2]]/M[pivots[:, 1], pivots[:, 2]])*M[pivots[:, 1], rand_pivots[: ,2]])))

end

function algorithm_2(M, pivots, num_rand_samples, iters, all_positions)
    
    # Pick random pivots
    existing_pivots = CartesianIndex.(pivots)
    available_positions = setdiff(all_positions, existing_pivots) # avoid picking random pivots which we already have saved in pivots
    rand_pivots_indices = rand(available_positions, num_rand_samples)
    rand_pivots = hcat(Int.(rand_pivots_indices .|> x -> x[1]), Int.(rand_pivots_indices .|> x -> x[2])) # just converts cartesian indices to a matrix

    # Compute the error of each random pivot
    error = compute_rand_pivots_error(M, rand_pivots, pivots)
    
    # Compute the final pivot star by iterating over the rows and then the columns an iters amount of times
    pivot_star = rand_pivots[argmax(error), :]
    for iter in 1:iters

        M_tilde = compute_approx_matrix(M, pivots)

        # Compute new i star
        error = abs.(M[:, pivot_star[2]] .- M_tilde[:, pivot_star[2]])
        tmp = argmax(error)
        if iter != 1 && pivot_star[1] == tmp # rook condition
            break
        end
        pivot_star[1] = tmp

        # Compute new j star
        pivot_star[2] = argmax(abs.(M[pivot_star[1], :] .- M_tilde[pivot_star[1], :]))
    
    end

    return pivot_star

end

function main(M, num_rand_samples, iters, num_pivots_to_add)

    plot_list = [heatmap(M, color = :viridis)]

    pivots = Array{Int64}(undef, num_pivots_to_add + 1, 2)
    pivots[1, :] = [argmax(M)[1], argmax(M)[2]]
    pivots_tmp = pivots[1:1, :]

    for i in 1:num_pivots_to_add

        println(i, ": ", compute_total_error(M, pivots_tmp))
        pivots[i+1, :] = algorithm_2(M, pivots_tmp, num_rand_samples, iters, all_positions)
        pivots_tmp = pivots[1:i+1, :]
        tmp = compute_approx_matrix(M, pivots_tmp)
        push!(plot_list, heatmap(tmp, color = :viridis))

    end

    plot_layout = @layout[grid(2, 3)]
    plot(plot_list[1:8:end]..., layout = plot_layout, colorbar = false, xticks = false, yticks = false)

end

pow = 8
num_rows, num_cols = 2^pow, 2^pow
all_positions = collect(CartesianIndices((num_rows, num_cols)))
M = [cos((i+j)/(num_cols/9))+sin((i*j)/(10*num_cols)) for i in 1:num_rows, j in 1:num_cols]
# img = load("/Users/takisangelides/Desktop/2.png")
# gray_img = Gray.(img)
# M = Float64.(imresize(gray_img, (num_rows, num_cols)))

num_rand_samples = 3
iters = 1000
num_pivots_to_add = 40

main(M, num_rand_samples, iters, num_pivots_to_add)
