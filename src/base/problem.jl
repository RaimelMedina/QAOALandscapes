abstract type AbstractProblem end

struct ClassicalProblem{T<:Real} <: AbstractProblem
    interactions::Dict{Vector{Int}, T}
    n::Int
end

nqubits(cp::ClassicalProblem)::Int = cp.n
Base.length(cp::ClassicalProblem) = length(cp.interactions)
locality(cp::ClassicalProblem) = maximum(map(length, keys(cp.interactions)))

function isZ2symmetric(cp::ClassicalProblem{T}) where T<:Real
    locality_of_hs = foldl(*, map(iseven ∘ length, keys(cp.interactions) |> collect))
    return locality_of_hs
end

function Base.show(io::IO, cp::ClassicalProblem{T}) where T<:Real
    str = "Classical problem on $(cp.n) qubits with interactions terms:"
    println(io, str)
    for k ∈ keys(cp.interactions)
        println(io, "├─ $(k) => $(cp.interactions[k])")
    end
    if isZ2symmetric(cp)
        println("Problem is Z₂ symmetric. Consider working in the corresponding parity sector")
    end
end

function ClassicalProblem(T::Type{<:Real}, g::SimpleGraph{<:Int})
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => T(1) for e in edges(g)) 
    return ClassicalProblem{T}(terms, nv(g))
end

function ClassicalProblem(g::SimpleWeightedGraph{<:Int, T}) where T<:Real
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => weight(e) for e in edges(g)) 
    return ClassicalProblem{T}(terms, nv(g))
end

function ClassicalProblem(interaction::Pair{Vector{Int}, T}, n::Int) where T<:Real
    terms = Dict(interaction)
    return ClassicalProblem{T}(terms, n)
end

function ClassicalProblem(interaction::Vector{Pair{Vector{Int}, T}}, n::Int) where T<:Real
    terms = Dict(interaction)
    return ClassicalProblem{T}(terms, n)
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
        if isZ2symmetric(cp)
            ham = zeros(Complex{T}, 2^(cp.n-1))
        else
            @info "Problem is not symmetric"
            ham = zeros(Complex{T}, 2^(cp.n))
        end
    else
        ham = zeros(Complex{T}, 2^(cp.n))
    end

    @inbounds for i in eachindex(ham)
        ham[i] = ham_value(i-1)
    end
    return ham
end
