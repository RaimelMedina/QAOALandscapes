# The problem with the initial implementation is that it roundes/truncates the keys of the dictionary
# which are the converged local minima at depth p. This then causes troubles when rolling down from this
# minima at circuit depth p+1

# I need to find a way to not modify the vector!!!

mutable struct IDGenerator
    idSeed::Int
    IDGenerator() = new(0)  # Constructor with initial idSeed set to 0
end

function generateId(generator::IDGenerator)
    id = generator.idSeed
    generator.idSeed += 1
    return id
end

mutable struct Node <: AbstractVector{Float64}
    parentId::Vector{Int64}
    id::Int64
    value::Vector{Float64}
end
 
Base.size(s::Node) = size(s.value)
Base.length(s::Node) = length(s.value)
Base.getindex(s::Node, i...) = getindex(s.value, i...)

function Base.show(io::IO, snode::Node)
    str = "Node with: \n" *
          "    parent ID = $(snode.parentId),\n" *
          "    ID = $(snode.id),\n" *
          "    number of layer = $(length(snode.value))"
    print(io, str)
end

mutable struct IdNodes
    parentId::Vector{Int64}
    id::Int64
end

function Base.show(io::IO, inode::IdNodes)
    str = "IdNode object with: \n" *
          "    parent ID = $(inode.parentId),\n" *
          "    ID = $(inode.id)"
    print(io, str)
end

function curateDict!(qaoa::QAOA, dict::Dict; rounding=true, digits=5)
    # first let's check that there are no repeated keys in here
    keys_of_dict = collect(keys(dict))
    energy_of_keys = qaoa.(keys_of_dict)
    energy_class_dict, _ = getEquivalentClasses(energy_of_keys; rounding=rounding, sigdigits=digits)
    
    temp_vec = [0]
    num_redundant = 0
    
    # go for all unique (rounded) energies
    for temp_vec in values(energy_class_dict)
        num_redundant = length(temp_vec)
        # if there is more than one parameter with the same energy
        # then we should keep one representative while throwing away 
        # the redundant parameters
        if num_redundant > 1
            for x in 2:num_redundant
                delete!(dict, keys_of_dict[temp_vec[x]])
            end
        end
    end
end

function keepKthBestMinima!(qaoa::QAOA, dict::Dict, k::Int; digits=5)
    num_keys = length(keys(dict))
    if num_keys ≤ k
        return nothing
    else
        keys_of_dict = collect(keys(dict))
        energy_of_keys = qaoa.(keys_of_dict)
        good_indexes = partialsortperm(energy_of_keys, 1:k)
        bad_indexes = setdiff(1:num_keys, good_indexes)
        
        for x in bad_indexes
            delete!(dict, keys_of_dict[x])
        end
        
        return dict
    end
end

function find_approx_key(dict::Dict{K, V}, key::K; digits=5) where {K, V}
    rounded_key = round.(key, digits=digits)
    for k in keys(dict)
        if isapprox(rounded_key, round.(k, digits=digits))
            return k 
        end
    end
    return nothing
end

function rollDownTSWithNode!(dict, qaoa::QAOA, node::Node, generator::IDGenerator; digits = 5, threaded=true)
    results = rollDown(qaoa, node.value, TSInitialization(), OptimizationOptimJL.BFGS())
    vecTemp = zeros(length(node) + 2) # temporal vector to store the keys
    
    for result in results
        for j in 1:2 # +/- direction
            candidate = result.params[:, j]
            vecTemp = find_approx_key(dict, candidate; digits=digits)
            if isnothing(vecTemp)
                dict[candidate] = IdNodes([node.id], generateId(generator))
            else
                push!(dict[vecTemp].parentId, node.id)
            end
        end
    end
end

function constructPartialOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; keep=5, digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]
    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()
    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        curateDict!(qaoa, dictNew)
        keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict
end

function constructOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]

    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()

    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        #curateDict!(qaoa, dictNew)
        #keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict

end

function getEdgesFromOptGraph(qaoa::QAOA, vec::Vector{Dict}; construct_graph=false)
    energy_data = Dict{Int, Float64}()
    edges_data  = Tuple{Int, Int}[]
    circ_depth  = Int[]
    for i ∈ eachindex(vec)
        for (k,v) in vec[i]
            push!(circ_depth, i)
            energy_data[v.id + 1] = qaoa(k)
            for uId in unique(v.parentId)
                if v.id != uId
                    if v.id < uId
                        push!(edges_data, (v.id + 1, uId + 1))
                    else
                        push!(edges_data, (uId + 1, v.id + 1))
                    end
                end
            end
        end
    end
    if construct_graph
        g = SimpleDiGraph(Edge{Int}.(edges_data))
        # for eg in edges_data
        #     if eg[1] != eg[2]
        #         add_edge!(g, eg[1], eg[2])
        #     end
        # end
        return g, edges_data, energy_data, circ_depth
    else
        return edges_data, energy_data, circ_depth
    end
end
# The problem with the initial implementation is that it roundes/truncates the keys of the dictionary
# which are the converged local minima at depth p. This then causes troubles when rolling down from this
# minima at circuit depth p+1

# I need to find a way to not modify the vector!!!

mutable struct IDGenerator
    idSeed::Int
    IDGenerator() = new(0)  # Constructor with initial idSeed set to 0
end

function generateId(generator::IDGenerator)
    id = generator.idSeed
    generator.idSeed += 1
    return id
end

mutable struct Node <: AbstractVector{Float64}
    parentId::Vector{Int64}
    id::Int64
    value::Vector{Float64}
end
 
Base.size(s::Node) = size(s.value)
Base.length(s::Node) = length(s.value)
Base.getindex(s::Node, i...) = getindex(s.value, i...)

function Base.show(io::IO, snode::Node)
    str = "Node with: \n" *
          "    parent ID = $(snode.parentId),\n" *
          "    ID = $(snode.id),\n" *
          "    number of layer = $(length(snode.value))"
    print(io, str)
end

mutable struct IdNodes
    parentId::Vector{Int64}
    id::Int64
end

function Base.show(io::IO, inode::IdNodes)
    str = "IdNode object with: \n" *
          "    parent ID = $(inode.parentId),\n" *
          "    ID = $(inode.id)"
    print(io, str)
end

function curateDict!(qaoa::QAOA, dict::Dict; rounding=true, digits=5)
    # first let's check that there are no repeated keys in here
    keys_of_dict = collect(keys(dict))
    energy_of_keys = qaoa.(keys_of_dict)
    energy_class_dict, _ = getEquivalentClasses(energy_of_keys; rounding=rounding, sigdigits=digits)
    
    temp_vec = [0]
    num_redundant = 0
    
    # go for all unique (rounded) energies
    for temp_vec in values(energy_class_dict)
        num_redundant = length(temp_vec)
        # if there is more than one parameter with the same energy
        # then we should keep one representative while throwing away 
        # the redundant parameters
        if num_redundant > 1
            for x in 2:num_redundant
                delete!(dict, keys_of_dict[temp_vec[x]])
            end
        end
    end
end

function keepKthBestMinima!(qaoa::QAOA, dict::Dict, k::Int; digits=5)
    num_keys = length(keys(dict))
    if num_keys ≤ k
        return nothing
    else
        keys_of_dict = collect(keys(dict))
        energy_of_keys = qaoa.(keys_of_dict)
        good_indexes = partialsortperm(energy_of_keys, 1:k)
        bad_indexes = setdiff(1:num_keys, good_indexes)
        
        for x in bad_indexes
            delete!(dict, keys_of_dict[x])
        end
        
        return dict
    end
end

function find_approx_key(dict::Dict{K, V}, key::K; digits=5) where {K, V}
    rounded_key = round.(key, digits=digits)
    for k in keys(dict)
        if isapprox(rounded_key, round.(k, digits=digits))
            return k 
        end
    end
    return nothing
end

function rollDownTSWithNode!(dict, qaoa::QAOA, node::Node, generator::IDGenerator; digits = 5, threaded=true)
    results = rollDown(qaoa, node.value, TSInitialization(), OptimizationOptimJL.BFGS())
    vecTemp = zeros(length(node) + 2) # temporal vector to store the keys
    
    for result in results
        for j in 1:2 # +/- direction
            candidate = result.params[:, j]
            vecTemp = find_approx_key(dict, candidate; digits=digits)
            if isnothing(vecTemp)
                dict[candidate] = IdNodes([node.id], generateId(generator))
            else
                push!(dict[vecTemp].parentId, node.id)
            end
        end
    end
end

function constructPartialOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; keep=5, digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]
    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()
    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        curateDict!(qaoa, dictNew)
        keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict
end

function constructOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]

    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()

    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        #curateDict!(qaoa, dictNew)
        #keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict

end

function getEdgesFromOptGraph(qaoa::QAOA, vec::Vector{Dict}; construct_graph=false)
    energy_data = Dict{Int, Float64}()
    edges_data  = Tuple{Int, Int}[]
    circ_depth  = Int[]
    for i ∈ eachindex(vec)
        for (k,v) in vec[i]
            push!(circ_depth, i)
            energy_data[v.id + 1] = qaoa(k)
            for uId in unique(v.parentId)
                if v.id != uId
                    if v.id < uId
                        push!(edges_data, (v.id + 1, uId + 1))
                    else
                        push!(edges_data, (uId + 1, v.id + 1))
                    end
                end
            end
        end
    end
    if construct_graph
        g = SimpleDiGraph(Edge{Int}.(edges_data))
        # for eg in edges_data
        #     if eg[1] != eg[2]
        #         add_edge!(g, eg[1], eg[2])
        #     end
        # end
        return g, edges_data, energy_data, circ_depth
    else
        return edges_data, energy_data, circ_depth
    end
end
module Experimental

using Graphs
using ProgressMeter
using ..QAOALandscapes
import ..QAOALandscapes:
    ClassicalProblem,
    QAOA,
    TSInitialization,
    Parameter,
    setvalue!,
    setRandomSeed,
    getEquivalentClasses,
    getInitialParameter,
    fourierOptimize,
    rollDown

using OptimizationOptimJL

include("data_wrapper.jl")
include("experimental.jl")

end
module Experimental

using Graphs
using ProgressMeter
using ..QAOALandscapes
import ..QAOALandscapes:
    ClassicalProblem,
    QAOA,
    TSInitialization,
    Parameter,
    setvalue!,
    setRandomSeed,
    getEquivalentClasses,
    getInitialParameter,
    fourierOptimize,
    rollDown

using OptimizationOptimJL

include("data_wrapper.jl")
include("experimental.jl")

end
# The problem with the initial implementation is that it roundes/truncates the keys of the dictionary
# which are the converged local minima at depth p. This then causes troubles when rolling down from this
# minima at circuit depth p+1

# I need to find a way to not modify the vector!!!


mutable struct IDGenerator
    idSeed::Int
    IDGenerator() = new(0)  # Constructor with initial idSeed set to 0
end

function generateId(generator::IDGenerator)
    id = generator.idSeed
    generator.idSeed += 1
    return id
end


mutable struct Node <: AbstractVector{Float64}
    parentId::Vector{Int64}
    id::Int64
    value::Vector{Float64}
end
 
Base.size(s::Node) = size(s.value)
Base.length(s::Node) = length(s.value)
Base.getindex(s::Node, i...) = getindex(s.value, i...)

function Base.show(io::IO, snode::Node)
    str = "Node with: \n" *
          "    parent ID = $(snode.parentId),\n" *
          "    ID = $(snode.id),\n" *
          "    number of layer = $(length(snode.value))"
    print(io,str)
end

mutable struct IdNodes
    parentId::Vector{Int64}
    id::Int64
end

function Base.show(io::IO, inode::IdNodes)
    str = "IdNode object with: \n" *
          "    parent ID = $(inode.parentId),\n" *
          "    ID = $(inode.id)"
    print(io,str)
end


function curateDict!(qaoa::QAOA, dict::Dict; rounding=true, digits=5)
    # first let's check that there are no repeated keys in here
    keys_of_dict = collect(keys(dict))
    energy_of_keys = qaoa.(keys_of_dict)
    energy_class_dict, _ = getEquivalentClasses(energy_of_keys; rounding=rounding, sigdigits=digits)
    
    temp_vec = [0]
    num_redundant = 0
    
    # go for all unique (rounded) energies
    for temp_vec in values(energy_class_dict)
        num_redundant = length(temp_vec)
        # if there is more than one parameter with the same energy
        # then we should keep one representative while throwing away 
        # the redundant parameters
        if num_redundant > 1
            for x in 2:num_redundant
                delete!(dict, keys_of_dict[temp_vec[x]])
            end
        end
    end
end

function keepKthBestMinima!(qaoa::QAOA, dict::Dict, k::Int; digits=5)
    num_keys = length(keys(dict))
    if num_keys ≤ k
        return nothing
    else
        keys_of_dict = collect(keys(dict))
        energy_of_keys = qaoa.(keys_of_dict)
        good_indexes = partialsortperm(energy_of_keys, 1:k)
        bad_indexes = setdiff(1:num_keys, good_indexes)
        
        for x in bad_indexes
            delete!(dict, keys_of_dict[x])
        end
        
        return dict
    end
end

function find_approx_key(dict::Dict{K, V}, key::K; digits=5) where {K, V}
    rounded_key = round.(key, digits=digits)
    for k in keys(dict)
        if isapprox(rounded_key, round.(k, digits=digits))
            return k 
        end
    end
    return nothing
end

function rollDownTSWithNode!(dict, qaoa::QAOA, node::Node, generator::IDGenerator; digits = 5, threaded=true)
    results = rollDown(qaoa, node.value, TSInitialization(), OptimizationOptimJL.BFGS())
    vecTemp = zeros(length(node) + 2) # temporal vector to store the keys
    
    for result in results
        for j in 1:2 # +/- direction
            candidate = result.params[:, j]
            vecTemp = find_approx_key(dict, candidate; digits=digits)
            if isnothing(vecTemp)
                dict[candidate] = IdNodes([node.id], generateId(generator))
            else
                push!(dict[vecTemp].parentId, node.id)
            end
        end
    end
end

function constructPartialOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; keep=5, digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]
    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()
    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        curateDict!(qaoa, dictNew)
        keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict
end

function constructOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; digits=5, threaded=true) where T<:Real
    @assert pmax ≥ 3
    generator = IDGenerator()
    node0 = Node([0], generateId(generator), Γ0)

    dict = Dict[]

    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()

    rollDownTSWithNode!(dict2, qaoa, node0, generator)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        
        @showprogress for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node, generator; digits=digits, threaded=threaded)
        end
        #curateDict!(qaoa, dictNew)
        #keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict

end

function getEdgesFromOptGraph(qaoa::QAOA, vec::Vector{Dict}; construct_graph=false)
    energy_data = Dict{Int, Float64}()
    edges_data  = Tuple{Int, Int}[]
    circ_depth  = Int[]
    for i ∈ eachindex(vec)
        for (k,v) in vec[i]
            push!(circ_depth, i)
            energy_data[v.id + 1] = qaoa(k)
            for uId in unique(v.parentId)
                if v.id != uId
                    if v.id < uId
                        push!(edges_data, (v.id + 1, uId + 1))
                    else
                        push!(edges_data, (uId + 1, v.id + 1))
                    end
                end
            end
        end
    end
    if construct_graph
        g = SimpleDiGraph(Edge{Int}.(edges_data))
        # for eg in edges_data
        #     if eg[1] != eg[2]
        #         add_edge!(g, eg[1], eg[2])
        #     end
        # end
        return g, edges_data, energy_data, circ_depth
    else
        return edges_data, energy_data, circ_depth
    end
end