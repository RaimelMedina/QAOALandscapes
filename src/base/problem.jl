struct ClassicalProblem{T<:Real} <: AbstractProblem
    interactions::Dict{Vector{Int}, T}
    n::Int
    locality::Vector{Int}
    degree::Union{Int, Nothing}
    weightedQ::Bool
end

Base.eltype(cp::ClassicalProblem{R}) where R<:Real = R
nqubits(cp::ClassicalProblem)::Int = cp.n
Base.length(cp::ClassicalProblem) = length(cp.interactions)
locality(cp::ClassicalProblem) = cp.locality

function locality(dict::Dict{Vector{Int}, T}) where T<:Real
    locality_of_hs = unique(map(length, keys(dict) |> collect))
    return locality_of_hs
end

function locality(pair::Pair{Vector{Int}, T}) where T<:Real
    return pair.first |> length
end

function locality(pairs::Vector{Pair{Vector{Int}, T}}) where T<:Real
    return map(locality, pairs) |> unique
end

function regularQ(cp::ClassicalProblem{R}) where R <: Real
    nint = length(cp)
    adjmat = zeros(Int, cp.n, nint)

    for (i, key) ∈ enumerate(keys(cp.interactions))
        for j ∈ eachindex(key)
            adjmat[key[j], i] = 1
        end
    end
    term_degrees = sum(adjmat, dims=2)
    println(adjmat)
    if allequal(term_degrees)
        return term_degrees[1]
    else
        return nothing
    end
end

function regularQ(interactions::Dict, n::Int; adjmat_return=false)
    nint = length(interactions)
    adjmat = zeros(Int, n, nint)

    for (i, key) ∈ enumerate(keys(interactions))
        for j ∈ eachindex(key)
            adjmat[key[j], i] = 1
        end
    end
    term_degrees = sum(adjmat, dims=2)
    if allequal(term_degrees)
        return term_degrees[1]
    else
        if adjmat_return
            return adjmat
        else
            return nothing
        end
    end
end

function Base.show(io::IO, cp::ClassicalProblem{T}) where T<:Real
    println(io, "Classical problem on $(cp.n) qubits")
    klocal = unique(cp.locality) |> sort
    if length(klocal)==1
        println(io, "With interaction terms of locality k=$(klocal[1])")
    else
        println(io, "With interaction terms of locality k=$(klocal)")
    end
    for k ∈ keys(cp.interactions)
        println(io, "├─ $(k) => $(cp.interactions[k])")
    end
    
end

function ClassicalProblem(T::Type{<:Real}, g::SimpleGraph{<:Int})
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => T(1) for e in edges(g))
    vertex_degree = degree(g)
    dg = allequal(vertex_degree) ? vertex_degree[1] : nothing
    locality_of_terms = [2]
    return ClassicalProblem{T}(
        terms, 
        nv(g), 
        locality_of_terms, 
        dg, 
        false
    )
end

function ClassicalProblem(g::SimpleWeightedGraph{<:Int, T}) where T<:Real
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => weight(e) for e in edges(g)) 
    vertex_degree = degree(g)
    dg = allequal(vertex_degree) ? vertex_degree[1] : nothing
    locality_of_terms = [2]
    return ClassicalProblem{T}(
        terms, 
        nv(g), 
        locality_of_terms, 
        dg, 
        true
    )
end

function ClassicalProblem(terms::Dict{Vector{Int}, T}, n::Int) where T<:Real
    isWeigted = !all(abs.(values(terms)) .== T(1))
    return ClassicalProblem{T}(
        terms, 
        n, 
        locality(terms), 
        regularQ(terms, n), 
        isWeigted
    )
end

function ClassicalProblem(interaction::Pair{Vector{Int}, T}, n::Int) where T<:Real
    terms = Dict(interaction)
    isWeighted = !(abs(interaction.second) == 1)
    return ClassicalProblem{T}(
        terms, 
        n, 
        locality(interaction), 
        nothing, 
        isWeighted
    )
end

function ClassicalProblem(interactions::Vector{Pair{Vector{Int}, T}}, n::Int) where T<:Real
    terms = Dict(interactions)
    return ClassicalProblem{T}(
        terms, 
        n, 
        locality(interactions), 
        regularQ(terms, n), 
        !foldl(&, abs.(values(terms)) .== 1)
    )
end

function ClassicalProblem(T::Type{<:Real}, mat::BitMatrix, J::Vector{Int})
    interactions = Dict{Vector{Int}, T}()
    @assert size(mat, 1) == length(J)
    N = size(mat, 2)
    
    for (i, h) in enumerate(eachrow(mat))
        interactions[findall(x->x==1, h)] = J[i] |> T
    end

    rsum = sum(mat, dims=1)
    zsum = sum(mat, dims=2)

    degree = allequal(rsum) ? rsum[1] : nothing
    locality_of_terms = unique(zsum)
    all_Js_equal = foldl(&, abs.(J) .== 1)

    return ClassicalProblem{T}(interactions, N, locality_of_terms, degree, !all_Js_equal)
end


function hamiltonian(cp::ClassicalProblem{T}, sym_sector = true) where T
    # Trying to avoid repeated allocations
    element_buffer = Vector{Bool}(undef, maximum(length(term) for term in keys(cp.interactions)))
    
    # Define ham_density_element to use the pre-allocated buffer
    function ham_density_element(x::Int, term::Vector{Int})
        # Reuse buffer instead of allocating in map
        for (idx, i) in enumerate(term)
            element_buffer[idx] = ((x >> (i-1)) & 1) == 1
        end
        idx = false
        @inbounds for i in 1:length(term)
            idx ⊻= element_buffer[i]
        end
        return Complex{T}(((-1)^idx) * cp.interactions[term])
    end
    
    z2_sym = foldl(&, iseven.(cp.locality))
    dim = if sym_sector && z2_sym
        2^(cp.n-1)
    else
        sym_sector && !z2_sym && @info "Problem is not symmetric"
        2^cp.n
    end
    ham = zeros(Complex{T}, dim)

    interaction_keys = collect(keys(cp.interactions))
    
    @inbounds for i in eachindex(ham)
        val = zero(Complex{T})
        for term in interaction_keys
            val += ham_density_element(i-1, term)
        end
        ham[i] = val
    end
    return ham
end

function Hc_ψ!(ham::Vector{S}, ψ::Vector{T}) where {S, T}
    @inbounds for i in eachindex(ψ)
        ψ[i] *= ham[i]
    end
    return nothing
end
