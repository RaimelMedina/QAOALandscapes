using QAOALandscapes
using LinearAlgebra
using ProgressMeter
using Random
using Graphs
using Plots
using JLD2

g = random_regular_graph(10, 3);
problem = ClassicalProblem(Float64, g)
qaoa = QAOA(problem)

# optimization graph #
Γinit, Einit = getInitialParameter(qaoa)
opt_graph = constructOptimizationGraph(qaoa, Γinit, 8);

edges_opt_graph, energ_opt_graph, depths_opt_graph = QAOALandscapes.getEdgesFromOptGraph(qaoa, opt_graph)

pwd()

jldsave(pwd()*"/examples/data_test_p_8.jld2"; edges=edges_opt_graph, vertexW=energ_opt_graph, depth=depths_opt_graph)

function warmOptimizeNewtonSaddles(qaoa::QAOA, p::Int, npoints::Int; seed=123)
    Random.seed!(seed)
    vector_of_Γs = rand(2p, npoints) * 2π
    
    data_energ = Float64[]
    data_param = Vector{Float64}[]

    @showprogress for x ∈ axes(vector_of_Γs, 2)
        temp = QAOALandscapes.modulatedNewtonSaddles(qaoa, vector_of_Γs[:, x])
        push!(data_param, temp[1])
        push!(data_energ, temp[2])
    end
    
    norm_points = map(x->norm(gradCostFunction(qaoa, x)), data_param)
    converged_idx = findall(x->x<1e-5, norm_points)

    converged_energies = data_energ[converged_idx]
    converged_params   = data_param[converged_idx]

    rounded_energies = round.(converged_energies, digits=6)
    idx = unique(i->rounded_energies[i], eachindex(rounded_energies))

    println("$(length(idx)) unique converged points out of $(npoints) initial particles")


    return converged_params[idx], converged_energies[idx]
end


params_p_2, energs_p_2 = warmOptimizeNewtonSaddles(qaoa, 2, 1000);
params_p_3, energs_p_3 = warmOptimizeNewtonSaddles(qaoa, 3, 1000; seed=456);
params_p_4, energs_p_4 = warmOptimizeNewtonSaddles(qaoa, 4, 1000);



energ_optgraph_p_2 = map(last, energ_opt_graph[depths_opt_graph .== 2])
energ_optgraph_p_3 = map(last, energ_opt_graph[depths_opt_graph .== 3]);
energ_optgraph_p_4 = map(last, energ_opt_graph[depths_opt_graph .== 4]);

function plot_converged_minima(param_graph, param_newton, title::String; sigdigits=6)
    pgraph  = round.(param_graph, sigdigits=sigdigits)
    pnewton = round.(param_newton, sigdigits=sigdigits)

    common_elements = intersect(pgraph, pnewton)

    index_graph  = findall(x -> x ∉ common_elements, pgraph)
    index_newton = findall(x -> x ∉ common_elements, pnewton)

    num_common_min = length(common_elements)
    plot_data = Plots.scatter(1:num_common_min, common_elements |> sort, label="Common minima", title=title)
    if length(index_graph) != 0
        Plots.scatter!(plot_data, num_common_min+1:num_common_min+length(index_graph), pgraph[index_graph] |> sort, label="Optimization graph")
    end
    Plots.scatter!(plot_data, num_common_min+1:num_common_min+length(index_newton), pnewton[index_newton] |> sort, label="Modulated Newton")
    return plot_data
end

plot_converged_minima(energ_optgraph_p_4, energs_p_4, "p = 4")
plot_converged_minima(energ_optgraph_p_3, energs_p_3, "p = 3")
plot_converged_minima(energ_optgraph_p_2, energs_p_2, "p = 2")

plot_p_3 = Plots.scatter(1:length(energs_p_3), energs_p_3 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=3 modulated Newton")
Plots.scatter!(plot_p_3, 1:length(energ_optgraph_p_3), energ_optgraph_p_3 |> sort, label="p=3 Optimization graph")

plot_p_4 = Plots.scatter(1:length(energs_p_4), energs_p_4 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=4 modulated Newton")
Plots.scatter!(plot_p_4, 1:length(energ_optgraph_p_4), energ_optgraph_p_4 |> sort, label="p=4 Optimization graph")

plot_p_2 = Plots.scatter(1:length(energs_p_2), energs_p_2 |> sort, markershape=:rect, ylabel="Energy", xlabel="point", label="p=2 modulated Newton")
Plots.scatter!(plot_p_2, 1:length(energ_optgraph_p_2), energ_optgraph_p_2 |> sort, label="p=2 Optimization graph")

Plots.savefig(plot_p_2, "p_2.pdf")
Plots.savefig(plot_p_3, "p_3.pdf")
Plots.savefig(plot_p_4, "p_4.pdf")