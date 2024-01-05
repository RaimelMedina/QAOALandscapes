using QAOALandscapes
using JLD2
using ProgressMeter
using SimpleWeightedGraphs
using Graphs


############# Reading data ###########
N     = parse(Int, ARGS[1])
pmax  = parse(Int, ARGS[2])
gpu   = parse(Int, ARGS[3])
######################################
seed = 123;
QAOALandscapes.setRandomSeed(seed)
#####################################

lendataSet = Dict(10=>18, 12=>34, 14=>55, 16=>40, 18=>40, 20=>40)
dir = "../data"

function load_graph_data(dir::String, N::Int)
    keyG   = "graph_list"
    fileG  = dir*"/unique_graphs_N_"*string(N)*".jld2"
    return jldopen(fileG)[keyG]
end

iter = Progress(lendataSet[N], desc="Collecting data for graphs of size N=$(N)...\n")

gpu==1 && println("Running simulations on the GPU")

for idx in 1:lendataSet[N]
    @show idx
    g = load_graph_data(dir, N)[idx]
    if gpu==0
        qaoa_data = QAOALandscapes.QAOAData(Float64, g, pmax)
    else
        qaoa_data = QAOALandscapes.QAOAData(METALBackend, Float32, g, pmax)
    end
    
    jldsave("data_N_$(N)_graph_$(idx)_pmax_$(pmax)_R_$(0).jld2"; qaoa_data)
    
    next!(iter; showvalues = [(:instance, idx)])
end