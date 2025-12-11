using HDF5
using Plots
using Statistics
using LaTeXStrings

function num_params_fraction_from_linkdims(l, N)

    res = 0
    res += 2*l[1]
    for (element_idx, element) in enumerate(l)
        if element_idx != length(l) && element_idx != 1
            res += element*2*l[element_idx-1]
        end
    end
    res += 2*l[end]

    return res/2^N

end

f1 = h5open("/Users/takisangelides/CFD_MPS/Burgers_Equation/velocity.h5", "r")
f2 = h5open("/Users/takisangelides/CFD_MPS/Burgers_Equation/bond_dimension.h5", "r")

N = 14
x_values = range(-7, 7, length = 2^N)

steps = maximum(parse.(Int, keys(f2)))

alphas = range(1.0, stop=0.4, length=steps)
p = plot(
    xlabel=L"x",
    ylabel=L"u(x)",
    title = "",
    legend=false,
    grid=true,
    xlims=(minimum(x_values), maximum(x_values)), # Adjust as needed
    ylims=(-1, 1), # Adjust as needed
    ticksize=5,
    background_color=:white
)
for step in 1:steps
    velocity = read(f1, "$(step)")
    plot!(p, x_values, velocity, alpha = alphas[step], color = :black)
end
savefig(p, "velocity.pdf")

D_max_list = []
num_params_list = []
for step in 1:steps
    bond_dimension = read(f2, "$(step)")
    push!(D_max_list, maximum(bond_dimension))
    push!(num_params_list, num_params_fraction_from_linkdims(bond_dimension, N))
end


p = plot(
    xlabel=L"t",
    ylabel=L"D",
    title = "",
    legend=false,
    grid=true,
    ticksize=5,
    background_color=:white
)
scatter!(p, D_max_list)
savefig(p, "bond_dimension.pdf")

p = plot(
    xlabel=L"t",
    ylabel=L"\textrm{Fraction}\quad\textrm{of}\quad\textrm{parameters}",
    title = "",
    legend=false,
    grid=true,
    ticksize=5,
    background_color=:white
)
scatter!(p, num_params_list)
savefig(p, "fraction_of_parameters.pdf")
