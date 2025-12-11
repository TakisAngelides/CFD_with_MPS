using LinearAlgebra
using ITensors
using Random

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

# ------------------------------------------------------------------------------------

# Implement the TT cross for a matrix m by n with just trivial choice for the rows and columns to evaluate

# m = 4
# n = 3
# D = 2

# A = rand(m, n)

# J_cols = 1:D

# I_rows = 1:D

# A_approx = A[:,J_cols] * inv(A[I_rows,J_cols]) * A[I_rows,:]

# println(norm(A - A_approx))

# ------------------------------------------------------------------------------------

# Implement algorithm 2 from Parallel cross interpolation for high-precision calculation of high-dimensional integrals

let

Random.seed!(1234)

N = 2 # Makes this a matrix to test algorithm 2 implementation
d = 30

A = rand(d.*ones(Int64, N)...)

function get_A_tilde(A, I, J, i, j)

    return (A[i, J]' * inv(A[I, J]) * A[I, j])[1]

end

function get_A_tilde_matrix(A, I, J)

    return A[:, J] * inv(A[I, J]) * A[I, :]

end

function algorithm_2!(I, J, A, num_samples, num_iter)

    num_rows, num_cols = size(A)
    zipped_IJ = zip(I, J)
    samples = []
    sample = [rand(1:num_rows), rand(1:num_cols)]
    while sample in zipped_IJ
        sample = [rand(1:num_rows), rand(1:num_cols)]
    end
    push!(samples, sample)    
    value = abs(A[samples[1]...] - get_A_tilde(A, I, J, samples[1]...))
    argmax_idx = 1
    for i in 2:num_samples
        sample = [rand(setdiff(1:num_rows, I)), rand(setdiff(1:num_cols, J))]
        while sample in samples || sample in zipped_IJ
            sample = [rand(setdiff(1:num_rows, I)), rand(setdiff(1:num_cols, J))]
        end
        push!(samples, sample)
        tmp = abs(A[samples[i]...] - get_A_tilde(A, I, J, samples[i]...))
        if value < tmp
            value = tmp
            argmax_idx = i
        end    
    end

    i_star, j_star = samples[argmax_idx]
    
    for _ in 1:num_iter

        tmp_list = []

        for sample in samples
            i = sample[1]
            push!(tmp_list, abs(A[i, j_star] - get_A_tilde(A, I, J, i, j_star)))
        end

        i_star = samples[argmax(tmp_list)][1]

        tmp_list = []

        for sample in samples
            j = sample[2]
            push!(tmp_list, abs(A[i_star, j] - get_A_tilde(A, I, J, i_star, j)))
        end

        j_star = samples[argmax(tmp_list)][2]

    end

    push!(I, i_star)
    push!(J, j_star)

    return I, J

end

I, J = [rand(1:d)], [rand(1:d)]

println(norm(get_A_tilde_matrix(A, I, J) - A))

num_samples = 10
num_iter = 10

for num in 1:29
    I, J = algorithm_2!(I, J, A, num_samples, num_iter)
    println(num, " ", norm(get_A_tilde_matrix(A, I, J) - A))
end

println(norm(get_A_tilde_matrix(A, I, J) - A))

end

