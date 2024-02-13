idSeed=0

function generateId()
    id = idSeed
    global idSeed = idSeed + 1
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
    str = "Node with: 
    parent ID = $(snode.parentId),
    ID = $(snode.id),
    number of layer = $(snode.length(value))"
    print(io,str)
end

mutable struct IdNodes
    parentId::Vector{Int64}
    id::Int64
end
function Base.show(io::IO, inode::IdNodes)
    str = "IdNode object with: 
    parent ID = $(inode.parentId),
    ID = $(inode.id)"
    print(io,str)
end


function curateDict!(qaoa::QAOA, dict::Dict; rounding=true, sigdigits=5)
    # first let's check that there are no repeated keys in here
    keys_of_dict = collect(keys(dict))
    energy_of_keys = qaoa.(keys_of_dict)
    _, energyEquivClasses = getEquivalentClasses(energy_of_keys; rounding=rounding, sigdigits=sigdigits)
    
    temp_vec = [0]
    num_redundant = 0
    
    # go for all unique (rounded) energies
    for k in keys(energyEquivClasses)
        temp_vec = energyEquivClasses[k]
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

function keepKthBestMinima!(qaoa::QAOA, dict::Dict, k::Int; sigdigits=5)
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

function rollDownTSWithNode!(dict, qaoa::QAOA, node::Node; sigdigits = 6, threaded=true)
    pdict, edict = rollDownTS(qaoa, node.value; threaded=true)
    
    vecTemp = similar(pdict["(1, 1)"][1])
    
    for k in keys(pdict)
        for j in 1:2
            vecTemp = round.(pdict[k][j], sigdigits=sigdigits)
            if haskey(dict, vecTemp)
                push!(dict[vecTemp].parentId, node.id)
            else
                dict[vecTemp] = IdNodes([node.id], generateId())
            end
        end
    end
end

# function constructPartialOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; keep=5, sigdigits=5) where T<:Real
#     @assert pmax ≥ 3
#     idSeed=1
#     node0 = Node([0], generateId(), round.(Γ0, sigdigits=sigdigits))

#     dict = Dict[]
#     push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

#     dict2 = Dict()
#     rollDownTSWithNode!(dict2, qaoa, node0)

#     push!(dict, dict2)

#     for p ∈ 3:pmax
#         println("Working with circuit depth p=$(p) -----")
#         dictNew = Dict()
#         for k in keys(dict[p-1])
#             node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
#             rollDownTSWithNode!(dictNew, qaoa, node)
#         end
#         curateDict!(qaoa, dictNew)
#         keepKthBestMinima!(qaoa, dictNew, keep)
#         push!(dict, dictNew)
#         println("Finished with p=$(p) -----")
        
#     end

#     return dict

# end

function constructOptimizationGraph(qaoa::QAOA, Γ0::Vector{T}, pmax::Int; sigdigits=5) where T<:Real
    @assert pmax ≥ 3
    idSeed=1
    node0 = Node([0], generateId(), round.(Γ0, sigdigits=sigdigits))

    dict = Dict[]
    push!(dict, Dict(node0.value => IdNodes(node0.parentId, node0.id)))

    dict2 = Dict()
    rollDownTSWithNode!(dict2, qaoa, node0)

    push!(dict, dict2)

    for p ∈ 3:pmax
        println("Working with circuit depth p=$(p) -----")
        dictNew = Dict()
        for k in keys(dict[p-1])
            node = Node(dict[p-1][k].parentId, dict[p-1][k].id, k)
            rollDownTSWithNode!(dictNew, qaoa, node)
        end
        #curateDict!(qaoa, dictNew)
        #keepKthBestMinima!(qaoa, dictNew, keep)
        push!(dict, dictNew)
        println("Finished with p=$(p) -----")
        
    end

    return dict

end