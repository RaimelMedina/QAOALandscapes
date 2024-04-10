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

vec_of_data = Array{QAOAData, 1}(undef, lendataSet[N])

for idx in 1:lendataSet[N]
    println("Working on instance $(idx) out of $(lendataSet[N])")
    g = load_graph_data(dir, N)[idx]
    if gpu==0
        vec_of_data[idx] = QAOAData(Float64, g, pmax)
    end
    next!(iter; showvalues = [(:instance, idx)])
end

jldsave("data_N_$(N)_pmax_$(pmax)_TS_1.jld2"; vec_of_data)