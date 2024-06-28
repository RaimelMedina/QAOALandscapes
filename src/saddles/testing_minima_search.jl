using QAOALandscapes
using LinearAlgebra
using ProgressMeter
using Random
using Graphs
using Plots

Random.seed!(123)
g = random_regular_graph(10, 3);
problem = ClassicalProblem(Float64, g)
qaoa = QAOA(problem)

# optimization graph #
Γinit, Einit = getInitialParameter(qaoa)

function warmOptimizeAdam(qaoa::QAOA, p::Int, npoints::Int)
    vector_of_Γs = rand(2p, npoints) * 2π
    
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x]) = QAOALandscapes.findMinimaAdam(qaoa, vector_of_Γs[:, x])
        next!(prog)
    end
    
    # norm_points = map(x->norm(gradCostFunction(qaoa, x)), data_param)
    # converged_idx = findall(x->x<1e-5, norm_points)

    # converged_energies = data_energ[converged_idx]
    # converged_params   = data_param[converged_idx]

    # rounded_energies = round.(converged_energies, sigdigits=5)
    # idx = unique(i->rounded_energies[i], eachindex(rounded_energies))

    # println("$(length(idx)) unique converged points out of $(npoints) initial particles")


    return vector_of_params, vector_of_energs
end


params_p_2, energs_p_2 = warmOptimizeAdam(qaoa, 3, 20_000)
# params_p_3, energs_p_3 = warmOptimizeNewton(qaoa, 3, 1500);
# params_p_4, energs_p_4 = warmOptimizeNewton(qaoa, 4, 1500)