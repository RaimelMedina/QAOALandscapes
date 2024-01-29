struct ClassicalProblem{T<:Real} <: AbstractProblem
    interactions::Dict{Vector{Int}, T}
    n::Int
    z2_sym::Bool
end

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
    return ClassicalProblem{T}(terms, nv(g), true)
end

function ClassicalProblem(g::SimpleWeightedGraph{<:Int, T}) where T<:Real
    terms = Dict{Vector{Int}, T}([e.src, e.dst] => weight(e) for e in edges(g)) 
    return ClassicalProblem{T}(terms, nv(g), true)
end

function ClassicalProblem(interaction::Pair{Vector{Int}, T}, n::Int) where T<:Real
    terms = Dict(interaction)
    return ClassicalProblem{T}(terms, n, z2SymmetricQ(interaction))
end

function ClassicalProblem(interactions::Vector{Pair{Vector{Int}, T}}, n::Int) where T<:Real
    terms = Dict(interactions)
    return ClassicalProblem{T}(terms, n, z2SymmetricQ(interactions))
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

    @inbounds for i in eachindex(ham)
        ham[i] = ham_value(i-1)
    end
    return ham
end

function Hc_ψ!(ham::AbstractVector{S}, ψ::AbstractVector{T}) where {S, T}
    dim = length(ψ)
    num_groups = dim ÷ MAX_THREADS

    @metal threads=MAX_THREADS groups=num_groups kernelHCψ!(ham, ψ)
    return nothing
end

function Hc_ψ!(ham::Vector{S}, ψ::Vector{T}) where {S, T}
    for i in eachindex(ψ)
        ψ[i] *= ham[i]
    end
    return nothing
end

# struct CostHamiltonian{P<:AbstractProblem, S<:AbstractVector}
#     problem::P
#     ham::S
    
#     function CostHamiltonian(::Type{<:MtlVector}, cp::ClassicalProblem{Float32})
#         ham = hamiltonian(cp)
#         P = typeof(cp)
#         H = typeof(ham)
#         return new{P, H}(cp, ham |> MtlArray)
#     end
#     function CostHamiltonian(cp::ClassicalProblem{R}) where R<:Real
#         ham = hamiltonian(cp)
#         P = typeof(cp)
#         H = typeof(ham)
#         return new{P, H}(cp, ham)
#     end
# end

function kernelHCψ!(hc::AbstractVector{T}, psi::AbstractVector{R}) where {T, R}
    i = thread_position_in_grid_1d()
    psi[i] *= hc[i]
    return
end

# function (cost_ham::CostHamiltonian{P, S<:MtlVector})(ψ::MtlVector{T}) where {P, S, T}
#     dim = length(ψ)
#     num_groups = dim ÷ MAX_THREADS

#     @metal threads=MAX_THREADS groups=num_groups kernelHCψ!(cost_ham.ham, psi)
# end

# function (cost_ham::CostHamiltonian{P, S})(ψ::Vector{T}) where {P, S, T}
#     for i in eachindex(ψ)
#         ψ[i] *= qaoa.HC[i]
#     end
#     return nothing
# end