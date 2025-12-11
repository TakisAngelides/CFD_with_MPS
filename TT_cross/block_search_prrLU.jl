function prrLU(B, τ, I, J, num_rows, num_cols)

    tmp = argmax(abs.(B))
    i, j = tmp[1], tmp[2]

    

    return I, J

end

function main(n_rook, τ, I, J, A, num_cols, num_rows)

    # Get a new random column that is not in J
    J_dash = vcat(J, rand(setdiff(1:num_cols, J)))

    # Set matrix B according to whether t is odd or even searching along the columns or rows respectively
    for t in 1:n_rook
        
        if t % 2 == 0
            B = A[I_dash, :] # I_dash is not defined above but the 't' for loop will always run the odd 't' case of the if statement first and below I_dash is defined
        else
            B = A[:, J_dash]
        end

        I_dash, J_dash = prrLU(B, τ, I, J, num_rows, num_cols)

        if I_dash == I && J_dash == J
            return I, J
        else
            I, J = I_dash, J_dash
        end

    end

end

# Create matrix for input
pow = 3
num_rows, num_cols = 2^pow, 2^pow
A = [cos((i+j)/(num_cols/9))+sin((i*j)/(10*num_cols)) for i in 1:num_rows, j in 1:num_cols]

# Initialize the pivot lists with 1 random row and column and make sure the function value at that pivot is not zero
I, J = [rand(1:num_rows)], [rand(1:num_cols)]
@assert(any(x -> abs(x) > 1e-14, A[I, J]), "The initial pivot should give at least something non-zero.")

# Initialize other input
τ = 1e-6
n_rook = 1

main(n_rook, τ, I, J, A, num_cols, num_rows)

println("Finished.")
