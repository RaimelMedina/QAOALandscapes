function fwht(a::T) where T <: AbstractVector
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

function ifwht(a::T) where T <: AbstractVector
    result = fwht(a)
    return result/length(a)
end

function fwht!(f::T, ldn::Int64) where T <: AbstractVector
    # Transform wrt. to Walsh-Kronecker basis (wak-functions).
    # Radix-2 decimation in time (DIT) algorithm.
    # Self-inverse.
    n = 1 << ldn
    for ldm in 1:ldn
        m = 1 << ldm
        mh = m >> 1
        for r in 0:m:n-m  # Corrected loop range calculation
            t1 = r
            t2 = r + mh
            for j in 0:(mh - 1)
                u = f[t1 + 1]  # Adding 1 as Julia uses 1-based indexing
                v = f[t2 + 1]  # Adding 1 as Julia uses 1-based indexing
                f[t1 + 1] = u + v  # Adding 1 as Julia uses 1-based indexing
                f[t2 + 1] = u - v  # Adding 1 as Julia uses 1-based indexing
                t1 += 1
                t2 += 1
            end
        end
    end
end

function ifwht!(a::T, ldn::Int64) where T <: AbstractVector
    fwht!(a, ldn)
    n  = 1 << ldn
    a .= a ./ n
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
    vec = zeros(Float64, n)
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

function isdRegularGraph(g::T, d::Int) where T <: AbstractGraph
    return reduce(*, degree(g) .== d)
end

function selectBestParams(eDict::Dict{String, Vector{Float64}}, pDict::Dict{String, Vector{Vector{Float64}}}, k::Int; sigdigits=6)
    # first, process the energies #
    kth_smallest_idx = findkthSmallestEnergy(reduce(hcat, values(eDict)), k; sigdigits = sigdigits)
    keysVec = keys(eDict) |> collect
    println("Keeping the $(k)-th best new local minima")
    return map(x->pDict[keysVec[x[2]]][x[1]], kth_smallest_idx)
end

function selectBestParams(eDict::Vector{Float64}, pDict::Vector{Vector{Float64}}, k::Int; sigdigits=6)
    # first, process the energies #
    kth_smallest_idx = findkthSmallestEnergy(eDict, k; sigdigits = sigdigits)
    println("Keeping the $(k)-th best new local minima")
    return map(x->pDict[x], kth_smallest_idx)
end

function findkthSmallestEnergy(arr::AbstractArray, k::Int; sigdigits=6)
    flattened = vec(arr)
    rounded_flattened = round.(flattened, sigdigits=sigdigits)

    unique_values_indices = Dict{Float64, Int}()
    for (index, value) in enumerate(rounded_flattened)
        if !haskey(unique_values_indices, value)
            unique_values_indices[value] = index
        end
    end

    unique_values = sort(collect(keys(unique_values_indices)))
    k_smallest_values = unique_values[1:k]

    k_smallest_indices = [unique_values_indices[v] for v in k_smallest_values]
    cartesian_indices = CartesianIndices(size(arr))
    row_col_indices = [cartesian_indices[i] for i in k_smallest_indices]

    return row_col_indices
end

