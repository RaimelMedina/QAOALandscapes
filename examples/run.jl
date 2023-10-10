using QAOALandscapes
using JLD2


############# Reading data ###########
N     = parse(Int, ARGS[1])
pmax  = parse(Int, ARGS[2])
R     = parse(Int, ARGS[3])
idx   = parse(Int, ARGS[4])
######################################


lendataSet = Dict(10=>18, 12=>34, 14=>55, 16=>60, 18=>60, 20=>60)
dir = "../data"

@assert idx ≤ lendataSet[N]

function load_graph_data(dir::String, N::Int)
    keyG   = "graph_list"
    fileG  = dir*"/unique_graphs_N_"*string(N)*".jld2"
    return jldopen(fileG)[keyG]
end

g = load_graph_data(dir, N)[idx]

qaoa_data = QAOALandscapes.QAOAData(g, pmax, R)

jldsave("data_N_$(N)_graph_$(idx)_pmax_$(pmax)_R_$(R).jld2"; qaoa_data)