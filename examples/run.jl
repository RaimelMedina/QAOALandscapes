using QAOALandscapes
using JLD2
using ProgressMeter
using SimpleWeightedGraphs
using Graphs


function toWeightedGraph(g::T, coeffs::Vector{R}) where {T<:SimpleGraph, R<:Real}
    n_vertices = nv(g)
    n_edges    = ne(g)
    @assert n_edges==length(coeffs)
    
    edge_set = collect(edges(g))
    
    sources = map(x->x.src, edge_set)
    destinations = map(x->x.dst, edge_set)
    
    return SimpleWeightedGraph(sources, destinations, coeffs)
    
end

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

weights_dict = jldopen(dir*"/weights.jld2")["weights_N_dict"]

iter = Progress(lendataSet[N], desc="Collecting data for graphs of size N=$(N)...")

gpu==1 && println("Running simulations on the GPU")

for idx in 1:lendataSet[N]
    @show idx
    g_unweighted = load_graph_data(dir, N)[idx]
    weights_new  = weights_dict[N][:, idx]
    if gpu==0
        @assert length(weights_new)==ne(g_unweighted)
        g = toWeightedGraph(g_unweighted, weights_new)
        qaoa_data = QAOALandscapes.QAOAData(Float64, g, pmax)
    else
        @assert length(weights_new)==ne(g_unweighted)
        g = toWeightedGraph(g_unweighted, Float32.(weights_new))
        qaoa_data = QAOALandscapes.QAOAData(METALBackend, Float32, g, pmax)
    end
    
    jldsave("weighted_data_N_$(N)_graph_$(idx)_pmax_$(pmax)_R_$(0).jld2"; qaoa_data)
    
    next!(iter; showvalues = [(:instance, idx)])
end