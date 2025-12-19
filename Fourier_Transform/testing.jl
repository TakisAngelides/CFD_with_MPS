using ITensors
using LinearAlgebra
using Plots
using QuanticsTCI
using TCIITensorConversion
using FFTW

# Implements QFT from https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.040318

h = [1 1; 1 -1] # missing factor of 1/sqrt(2) out front is to match the convention of fourier transform between fft and qft
ITensors.op(::OpName"h",::SiteType"S=1/2") = h

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

function get_hadamard_mpo(sites, idx)

    opsum = OpSum()
    opsum += h,idx
    return MPO(opsum, sites)

end

function get_qft_mpo(sites, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, 1)
    for i in 1:N-1
        mpo = apply(get_phase_mpo(sites, i, 1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i + 1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

end

function get_sin_mps(sites, a, b, dx)

    """
    This function gets specifically the MPS = a * sin(b * x)
    """
    
    mps = MPS(sites)
    N = length(sites)
    links = [Index(2, "Link,l=$(n)") for n in 1:N-1]

    for n in 1:N

        if n == 1

            s, lr = sites[n], links[n]
            
            mps[n] = ITensor(ComplexF64, s, lr)

            mps[n][s => 1, lr => 1] = a/(2*1im)
            mps[n][s => 2, lr => 1] = a*exp(1im*b*2^(N-n)*dx)/(2*1im)
            
            mps[n][s => 1, lr => 2] = -a/(2*1im)
            mps[n][s => 2, lr => 2] = -a*exp(-1im*b*2^(N-n)*dx)/(2*1im)

        elseif n == N

            s, ll = sites[n], links[n-1]
            
            mps[n] = ITensor(ComplexF64, s, ll)

            mps[n][s => 1, ll => 1] = 1
            mps[n][s => 2, ll => 1] = exp(1im*b*2^(N-n)*dx)

            mps[n][s => 1, ll => 2] = 1
            mps[n][s => 2, ll => 2] = exp(-1im*b*2^(N-n)*dx)

        else

            s, ll, lr = sites[n], links[n-1], links[n]

            mps[n] = ITensor(ComplexF64, s, ll, lr)

            mps[n][s => 1, ll => 1, lr => 1] = 1
            mps[n][s => 2, ll => 1, lr => 1] = exp(1im*b*2^(N-n)*dx)

            mps[n][s => 1, ll => 2, lr => 2] = 1
            mps[n][s => 2, ll => 2, lr => 2] = exp(-1im*b*2^(N-n)*dx)

        end
    end

    return mps

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
    for i in 1:2^N
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        if reverse_flag
            binary_list = reverse(decimal_to_padded_binary_list(i-1, N))
        else
            println(i-1, " : ", decimal_to_padded_binary_list(i-1, N))
            binary_list = decimal_to_padded_binary_list(i-1, N) 
        end
        push!(res, tmp[binary_list...])
    end
    return res

end

function mpo_to_matrix(mpo)

    N = length(mpo)
    res = contract(mpo)
    res = Array(res, inds(res; :plev => 1)..., inds(res; :plev => 0)...)
    res = reshape(res, 2^N, 2^N)

    return res

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

function get_inv_qft_mpo(sites, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, N)
    for i in N-1:-1:1
        mpo = apply(get_phase_mpo(sites, i, -1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

end

N = 8
dx = 1/(2^N-1)
cutoff = 1e-16
x_values = range(-1, 1; length = 2^N)
maxbonddim = 100 
tolerance = 1e-8 

max_sweeps = 100
tol = 1e-10

f = 1
sites, mps = tt_cross(x -> sin(2*pi*f*x), x_values, maxbonddim, tolerance)
# sites, mps = tt_cross(x -> exp(-x^2), x_values, maxbonddim, tolerance)
# sites, mps = tt_cross(x -> sinc(x), x_values, maxbonddim, tolerance)
# sites, mps = tt_cross(x -> 1, x_values, maxbonddim, tolerance)
mps_list = mps_to_list(mps)

qft_mpo = get_qft_mpo(sites, cutoff) # applies Q part of figure 1

qft_mps = apply(qft_mpo, mps)
q_mps_list = []
mps_reversed = MPS(sites)
for i in 1:N
    mps_reversed[i] = qft_mps[N-i+1]
end
qft_mps_list = mps_to_list(mps_reversed; reverse_flag = false) # the reverse flag implements the R part in figure 1

fft_mps_list = fft(mps_list)

p0 = plot(x_values, mps_list, title="Original Function", legend=false)
display(p0)

p1 = plot(abs.(qft_mps_list), title="QFT Magnitude", legend=false)
p2 = plot(-angle.(qft_mps_list), title="QFT Phase", legend=false, marker = "o") # minus sign here is to match the convention of fourier transform between fft and qft
p3 = plot(abs.(fft_mps_list), title = "FFT Magnitude", legend=false)
p4 = plot(angle.(fft_mps_list), title = "FFT Phase", marker = "o", legend=false)

plot(p1, p2, p3, p4, layout = (2, 2), size=(800, 400), legend=false)
