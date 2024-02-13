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

getWeight(T::Type{<:Real}, edge::T1) where T1<:Graphs.SimpleGraphs.SimpleEdge = T(1)
getWeight(T::Type{<:Real}, edge::T1) where T1<:SimpleWeightedEdge = T(edge.weight)

function whichTSType(s::String)
    vec = parse.(Int, split(s, ['(', ',', ')'])[2:3])
    tsType = (vec[1]==vec[2]) ? "symmetric" : "non_symmetric"
    return vec[1], tsType
end

function _onehot(T::Type{<:Number}, i::Int, n::Int)
    (i>n || i<1) ? throw(ArgumentError("Wrong indexing of the unit vector. Please check!")) : nothing
    vec = zeros(T, n)
    vec[i] = T(1)
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

function interpolateParams(Γ::Vector{T}) where T <: Real
    p = length(Γ) ÷ 2

    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]

    itp_γ = cubic_spline_interpolation(1:p, γ)
    itp_β = cubic_spline_interpolation(1:p, β)

    return itp_γ, itp_β
end

function getSmoothnessOfCurve(Γ::Vector{T}) where T<:Real
    intp_γ, intp_β = interpolateParams(Γ)
    fγ(x::T) where T<:Real = abs2(Interpolations.hessian(intp_γ, x)[1])
    fβ(x::T) where T<:Real = abs2(Interpolations.hessian(intp_β, x)[1])

    p  = Float64(length(Γ) ÷ 2)
    p0 = 1.0
    Iγ, _ = quadgk(fγ, p0, p)
    Iβ, _ = quadgk(fγ, p0, p)

    return Iγ, Iβ
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

"""
    getEquivalentClasses(vec::Vector{T}; sigdigits = 5) where T <: Real

Computes the equivalence classes of the elements of the input vector `vec`
The elements are rounded to a certain number of significant digits (default is 5) to group the states with approximately equal values.

# Arguments
* `vec::Vector{T<:Real}`
* `sigdigits=5`: Significant digits to which energies are rounded.

# Returns
* `data_states::Dict{Float64, Vector{Int}}`: Keys are unique elements (rounded) and values corresponds to the index of elements with the same key.
"""
function getEquivalentClasses(vec::Vector{T}; rounding=false, sigdigits = 5) where T <: Real
    println("Rounding of the elements is set to $(rounding)")
    dictUniqueElements = Dict{T, Vector{Int}}()
    temp_element = T(0)
    if rounding
        for (i, elem) in enumerate(vec)
            temp_element = round(elem, sigdigits=sigdigits)
            if haskey(dictUniqueElements, temp_element)
                push!(dictUniqueElements[temp_element], i)
            else
                dictUniqueElements[temp_element] = [i]
            end
        end
    else
        for (i, elem) in enumerate(vec)
            temp_element = elem
            if haskey(dictUniqueElements, temp_element)
                push!(dictUniqueElements[temp_element], i)
            else
                dictUniqueElements[temp_element] = [i]
            end
        end
    end
    data = collect(values(dictUniqueElements))[sortperm(collect(keys(dictUniqueElements)))]
    return dictUniqueElements, data 
end


function hc2Terms(T::Type{<:Real}, qaoa::QAOA{T1, T2, T3}) where {T1, T2, T3}
    edge_list = collect(keys(qaoa.problem.interactions))
    dict2LocalTerms = Dict{Vector{Int}, T}()
    dict4LocalTerms = Dict{Vector{Int}, T}()
    weight_list = values(qaoa.problem.interactions) |> collect
    
    val_temp = Integer[]
    weight_val = T(0)
    id_counter = T(0)
    
    for (j, e1) in enumerate(edge_list)
        for (i, e2) in enumerate(edge_list)
            val_temp = symdiff(e1, e2) |> sort
            
            weight_val = weight_list[j]*weight_list[i]
            
            if length(val_temp) == 2
                if haskey(dict2LocalTerms, val_temp)
                    dict2LocalTerms[val_temp] += weight_val
                else
                    dict2LocalTerms[val_temp] = weight_val
                end
            elseif length(val_temp)==4
                if haskey(dict4LocalTerms, val_temp)
                    dict4LocalTerms[val_temp] += weight_val
                else
                    dict4LocalTerms[val_temp] = weight_val
                end
            else
                id_counter += weight_val
            end
        end
    end
    return id_counter, dict2LocalTerms, dict4LocalTerms
end


struct TaylorTermsTS{P, H, M, T}
    qaoa::QAOA{P, H, M}
    T0::T
    T2h::H
    T4h::H
end

function TaylorTermsTS(T::Type{<:Real}, qaoa::QAOA{T1, T2, T3}) where {T1, T2, T3}
    id, dict_T2, dict_T4 = hc2Terms(T, qaoa)
    T2h  = hamiltonian(ClassicalProblem(dict_T2, qaoa.N))
    T4h  = hamiltonian(ClassicalProblem(dict_T4, qaoa.N))
   
    @assert real((T2h + T4h) .+ id) ≈ real(qaoa.HC .^ 2)
    return TaylorTermsTS{T1, T2, T3, T}(
        qaoa,
        id,
        T2h,
        T4h
    )
end

function T2Energy(taylor::TaylorTermsTS, Γ::Vector{T}) where T<:Real
    ψ   = getQAOAState(taylor.qaoa, Γ)
    ψT2 = UT2ψ0(taylor, Γ)
    return dot(ψ, taylor.qaoa.HC .* ψT2)
end

function T4Energy(taylor::TaylorTermsTS, Γ::Vector{T}) where T<:Real
    ψ   = getQAOAState(taylor.qaoa, Γ)
    ψT4 = UT4ψ0(taylor, Γ)
    return dot(ψ, taylor.qaoa.HC .* ψT4)
end

function UT2ψ0(taylor::TaylorTermsTS, Γ::Vector{T}) where T<:Real
    #z = Complex{T}(2^(- taylor.qaoa.N / 2))
    z = copy(
        taylor.qaoa.initial_state
    )
    return getQAOAState(taylor.qaoa, Γ, z .* taylor.T2h)
end

function UT4ψ0(taylor::TaylorTermsTS, Γ::Vector{T}) where T<:Real
    z = copy(
        taylor.qaoa.initial_state
    )
    return getQAOAState(taylor.qaoa, Γ, z .* taylor.T4h)
end

function UHCψ0(qaoa::QAOA, Γ::Vector{T}) where T<:Real
    z = copy(
        qaoa.initial_state
    )
    return getQAOAState(qaoa, Γ, z .* qaoa.HC)
end

function UHC2ψ0(qaoa::QAOA, Γ::Vector{T}) where T<:Real
    z = Complex{T}(2^(-qaoa.N / 2))
    return getQAOAState(qaoa, Γ, z .* (qaoa.HC .^ 2))
end

function UHC2ψ0(taylor::TaylorTermsTS, Γ::Vector{T}) where T<:Real
    z = copy(
        taylor.qaoa.initial_state
    )
    return getQAOAState(taylor.qaoa, Γ, z .* (taylor.qaoa.HC .^ 2))
end

function UOϵψ0(taylor::TaylorTermsTS, Γ::Vector{T}, ϵ::T) where {T<:Real}
    c2  = Complex{T}(exp(im * 2 * sqrt(2) * ϵ) - 1)
    c4  = Complex{T}(exp(im * 4 * sqrt(2) * ϵ) - 1)

    value = c2 * UT2ψ0(taylor, Γ) + c4 * UT4ψ0(taylor, Γ)
    return value
end