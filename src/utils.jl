function fwht(a::Vector{T}) where T
    h   = 1
    dim = length(a) 
    tmp = copy(a)
    while 2 * h <= dim
        for i ∈ 0:2h:dim-2
            for j ∈ i:(i + h - 1)
                x = tmp[j + 1]
                y = tmp[j + h + 1]
                tmp[j + 1] = x + y
                tmp[j + h + 1] = x - y
            end
        end
        h *= 2
    end
    return tmp
end

function ifwht(a::Vector{T}) where T
    return fwht(a) / length(a)
end

getWeight(edge::T) where T<:Graphs.SimpleGraphs.SimpleEdge = 1.0
getWeight(edge::T) where T<:SimpleWeightedEdge = edge.weight

function gradStdTest(v::Vector{Float64})
    dim = length(v)
    Δβ = v[2:2:dim] |> diff
    Δγ = v[1:2:dim] |> diff
    return mean([std(Δβ), std(Δγ)])
end

function selectSmoothParameter(Γ1::Vector{Float64}, Γ2::Vector{Float64})
    vectors = [Γ1, Γ2]
    res = gradStdTest.(vectors)
    idx = argmin(res)
    return idx, vectors[idx]
end

function selectSmoothParameter(Γ::Vector{Vector{Float64}})
    res = gradStdTest.(Γ)
    idx = argmin(res)
    return idx, Γ[idx]
end

function whichTSType(s::String)
    vec = parse.(Int, split(s, ['(', ',', ')'])[2:3])
    tsType = (vec[1]==vec[2]) ? "symmetric" : "non_symmetric"
    return vec[1], tsType
end

function _onehot(i::Int, n::Int)
    (i>n || i<1) ? throw(ArgumentError("Wrong indexing of the unit vector. Please check!")) : nothing
    vec = zeros(Int64, n)
    vec[i] = 1
    return vec
end

@doc raw"""
    spinChain(n::Int; bcond="pbc")

Constructs the graph for the classical Ising Hamiltonian on a chain with
periodic boundary conditions determined by the *keyword* argument `bcond`
"""
function spinChain(n::Int; bcond="pbc")
    latticeGraph = path_graph(n)
    if bcond == "pbc"
        add_edge!(latticeGraph, 1, n)
    end
    return latticeGraph
end

function parity_of_integer(x::Integer)
    parity = 0
    while x != 0
        parity ⊻= x & 1
        x >>= 1
    end
    parity
end

function isdRegularGraph(g::Graph, d::Int)
    adjMat     = adjacency_matrix(g)
    regularity = map(x->length(findall(!iszero, x)), eachcol(adjMat))
    boolVector = regularity .== d
    return boolVector==ones(size(adjMat)[1])
end