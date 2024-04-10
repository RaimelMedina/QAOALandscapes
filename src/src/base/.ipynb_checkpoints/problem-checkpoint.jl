struct ClassicalProblem{T<:Real} <: AbstractProblem
    interactions::Dict{Vector{Int}, T}
    n::Int
    z2_sym::Bool
    degree::Union{Int, Nothing}
    weightedQ::Bool
end

Base.eltype(cp::ClassicalProblem{R}) where R<:Real = R
nqubits(cp::ClassicalProblem)::Int = cp.n
Base.length(cp::ClassicalProblem) = length(cp.interactions)
locality(cp::ClassicalProblem) = maximum(map(length, keys(cp.interactions)))

function z2SymmetricQ(cp::ClassicalProblem{T}) where T<:Real
    return cp.z2_sym
end

function z2SymmetricQ(dict::Dict{Vector{Int}, T}) where T<:Real
    locality_of_hs = foldl(*, map(iseven ∘ length, keys(dict) |> collect))
    return locality_of_hs
end

function z2SymmetricQ(pair::Pair{Vector{Int}, T}) where T<:Real
    return pair.first |> length |> iseven
end

function z2SymmetricQ(pairs::Vector{Pair{Vector{Int}, T}}) where T<:Real
    return foldl(*, map(z2SymmetricQ, pairs))
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
    if cp.z2_sym
        str = "Z₂ symmetric classical problem on $(cp.n) qubits with interactions terms:"
    else
        str = "Classical problem on $(cp.n) qubits with interactions terms:"
    end
    println(io, str)
    for k ∈ keys(cp.interactions)
        println(io, "├─ $(k) => $(cp.interactions[k])")
    end
end

function ClassicalProblem(T::Type{<:Real}, g::SimpleGraph{<:Int})
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => T(1) for e in edges(g))
    vertex_degree = degree(g)
    dg = allequal(vertex_degree) ? vertex_degree[1] : nothing 
    return ClassicalProblem{T}(terms, nv(g), true, dg, false)
end

function ClassicalProblem(g::SimpleWeightedGraph{<:Int, T}) where T<:Real
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => weight(e) for e in edges(g)) 
    vertex_degree = degree(g)
    dg = allequal(vertex_degree) ? vertex_degree[1] : nothing 
    return ClassicalProblem{T}(terms, nv(g), true, dg, true)
end

function ClassicalProblem(interaction::Pair{Vector{Int}, T}, n::Int) where T<:Real
    terms = Dict(interaction)
    return ClassicalProblem{T}(terms, n, z2SymmetricQ(interaction), nothing, true)
end

function ClassicalProblem(interactions::Vector{Pair{Vector{Int}, T}}, n::Int) where T<:Real
    terms = Dict(interactions)
    return ClassicalProblem{T}(terms, n, 
    z2SymmetricQ(interactions), 
    regularQ(terms, n), 
    allequal(values(terms))
    )
end

function ClassicalProblem(terms::Dict{Vector{Int}, T}, n::Int) where T<:Real
    return ClassicalProblem{T}(terms, n, 
    z2SymmetricQ(terms), 
    regularQ(terms, n), 
    allequal(values(terms))
    )
end

function ClassicalProblem(T::Type{<:Real}, mat::BitMatrix, J::BitArray)
    interactions = Dict{Vector{Int}, T}()
    @assert size(mat, 1) == length(J)
    N = size(mat, 2)
    
    for (i, h) in enumerate(eachrow(mat))
        interactions[findall(x->x==1, h)] = J[i] |> T
    end

    rsum = sum(mat, dims=1)
    zsum = sum(mat, dims=2)

    degree = allequal(rsum) ? rsum[1] : nothing
    z2     = allequal(iseven.(zsum)) ? true : false
    weight = allequal(J)

    return ClassicalProblem{T}(interactions, N, z2, degree, weight)
end


function hamiltonian(cp::ClassicalProblem{T}, sym_sector = true) where T
    function ham_density_element(x::Int, term::Vector{Int})
        elements = map(i->((x>>(i-1))&1), term)
        idx      = foldl(⊻, elements)
        val      = Complex{T}(((-1)^idx)*cp.interactions[term])
        return val
    end

    function ham_value(x::Int)
        return sum(k->ham_density_element(x, k), keys(cp.interactions))
    end
    
    if sym_sector
        if z2SymmetricQ(cp)
            ham = zeros(Complex{T}, 2^(cp.n-1))
        else
            @info "Problem is not symmetric"
            ham = zeros(Complex{T}, 2^(cp.n))
        end
    else
        ham = zeros(Complex{T}, 2^(cp.n))
    end

    for i in eachindex(ham)
        ham[i] = ham_value(i-1)
    end
    return ham
end

function Hc_ψ!(ham::Vector{S}, ψ::Vector{T}) where {S, T}
    for i in eachindex(ψ)
        ψ[i] *= ham[i]
    end
    return nothing
end
