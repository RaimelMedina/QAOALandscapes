using QAOALandscapes
using LinearAlgebra
using ProgressMeter
using Random
using Graphs
using Plots

g = random_regular_graph(10, 3);
problem = ClassicalProblem(Float64, g)
qaoa = QAOA(problem)

# optimization graph #
Γinit, Einit = getInitialParameter(qaoa)
opt_graph = constructOptimizationGraph(qaoa, Γinit, 4);

_, energ_opt_graph, depths_opt_graph = QAOALandscapes.getEdgesFromOptGraph(qaoa, opt_graph)

function warmOptimizeNewton(qaoa::QAOA, p::Int, npoints::Int; seed=123) where T<:AbstractFloat
    Random.seed!(seed)
    vector_of_Γs = rand(2p, npoints) * 2π
    
    data_energ = Float64[]
    data_param = Vector{Float64}[]

    @showprogress for x ∈ axes(vector_of_Γs, 2)
        temp = QAOALandscapes.modulatedNewton(qaoa, vector_of_Γs[:, x])
        push!(data_param, temp[1])
        push!(data_energ, temp[2])
    end
    
    norm_points = map(x->norm(gradCostFunction(qaoa, x)), data_param)
    converged_idx = findall(x->x<1e-5, norm_points)

    converged_energies = data_energ[converged_idx]
    converged_params   = data_param[converged_idx]

    rounded_energies = round.(converged_energies, sigdigits=5)
    idx = unique(i->rounded_energies[i], eachindex(rounded_energies))

    println("$(length(idx)) unique converged points out of $(npoints) initial particles")


    return converged_params[idx], converged_energies[idx]
end




params_p_2, energs_p_2 = warmOptimizeNewton(qaoa, 2, 500)
params_p_3, energs_p_3 = warmOptimizeNewton(qaoa, 3, 1000);
params_p_4, energs_p_4 = warmOptimizeNewton(qaoa, 4, 1000)

energ_optgraph_p_2 = map(last, energ_opt_graph[depths_opt_graph .== 2])
energ_optgraph_p_3 = map(last, energ_opt_graph[depths_opt_graph .== 3]);
energ_optgraph_p_4 = map(last, energ_opt_graph[depths_opt_graph .== 4]);


plot_p_3 = Plots.scatter(1:length(energs_p_3), energs_p_3 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=3 modulated Newton")
Plots.scatter!(plot_p_3, 1:length(energ_optgraph_p_3), energ_optgraph_p_3 |> sort, label="p=3 Optimization graph")

plot_p_4 = Plots.scatter(1:length(energs_p_4), energs_p_4 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=4 modulated Newton")
Plots.scatter!(plot_p_4, 1:length(energ_optgraph_p_4), energ_optgraph_p_4 |> sort, label="p=4 Optimization graph")

plot_p_2 = Plots.scatter(1:length(energs_p_2), energs_p_2 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=2 modulated Newton")
Plots.scatter!(plot_p_2, 1:length(energ_optgraph_p_2), energ_optgraph_p_2 |> sort, label="p=2 Optimization graph")

Plots.savefig(plot_p_2, "p_2.pdf")
Plots.savefig(plot_p_3, "p_3.pdf")
Plots.savefig(plot_p_4, "p_4.pdf")