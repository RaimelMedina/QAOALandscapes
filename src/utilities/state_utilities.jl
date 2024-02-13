function plus_state(T::Type{<:Real}, N::Int)
    return fill(Complex{T}(1/sqrt(1<<N)), 1<<N)
end

"""
    getStateProjection(qaoa::QAOA, params, stateIndex::Vector{Int64})

Calculates the projection of the QAOA state onto the state subspace determined by `gsIndex`. It also returns the corresponding orthogonal complement. 
The QAOA state is determined by the given parameters `params`.

# Arguments
* `qaoa::QAOA`: QAOA object.
* `params`: Parameters determining the QAOA state.
* `stateIndex::Vector{Int64}`: Indices of the ground state components.

# Returns
* `normState`: Norm of the projection of the QAOA state onto the state subspace given by `stateIndex`.
* `normState_perp`: Norm of the projection of the QAOA state onto the orthogonal complement of the specificied state subspace.
* `ψIndex`: Normalized projection of the QAOA state onto the state subspace.
* `ψIndex_perp`: Normalized projection of the QAOA state onto the orthogonal complement of the state subspace.
"""
function getStateProjection(qaoa::QAOA{P, H, M}, params::Vector{T}, gsIndex::Vector{Int}) where {P, H, M, T<:Real}
    ψMin  = getQAOAState(qaoa, params)
    ψ = sum(map(x->_onehot(Complex{T}, x, 1<<qaoa.N)*ψMin[x], gsIndex))[:]

    normState = norm(ψ)
    normalize!(ψ)

    ψ_perp    = ψMin - dot(ψ, ψMin)*ψ
    normState_perp = norm(ψ_perp)
    normalize!(ψ_perp)

    return normState, normState_perp, ψ, ψ_perp
end

function getStateProjection(qaoa::QAOA{P, H, M}, ψinit::AbstractVector{T}, gsIndex::Vector{Int}) where {P, H, M, T}
    ψMin  = ψinit
    ψ = sum(map(x->_onehot(T, x, 1<<qaoa.N)*ψMin[x], gsIndex))[:]

    normState = norm(ψ)
    normalize!(ψ)

    ψ_perp    = ψMin - dot(ψ, ψMin)*ψ
    normState_perp = norm(ψ_perp)
    normalize!(ψ_perp)

    return normState, normState_perp, ψ, ψ_perp
end

"""
    computationalBasisWeights(ψ, equivClasses)

Computes the computational basis weights for a given state vector `ψ` according to the provided equivalence classes, by summing up the squared magnitudes of the elements with the same equivalence class.
Here `ψ` and `equivClasses` are ment to live in the same Hilbert space basis. 

# Arguments
* `ψ`: The state vector.
* `equivClasses`: A Vector of Vectors, where each inner Vector contains the indices of the elements belonging to the same equivalence class.

# Returns
* `basis_weights::Vector{Float64}`: A Vector containing the summed squared magnitudes of `ψ` for each equivalence class.
"""
function computationalBasisWeights(ψ, equivClasses)
    return map(x-> sum(abs2.(getindex(ψ, x))), equivClasses)
end

function gsFidelity(qaoa::QAOA{P, H, M}, Γ::AbstractVector{T}, gsIndex::Vector{Int}) where {P, H, M, T<:Real}
    return computationalBasisWeights(getQAOAState(qaoa, Γ), gsIndex) |> sum
end

function timeToSolution(qaoa::QAOA{P, H, M}, 
    Γ::Vector{T},
    gsIndex::Vector{Int}; 
    pd=T(0.99)) where {P, H, M, T <:Real}

    @debug "Here we are assuming that the cost Hamiltonian is classical!"
    p = length(Γ) ÷ 2
    
    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]

    total_time = sum(abs.(γ) + abs.(β))
    
    ψ = getQAOAState(qaoa, Γ)

    pgs = computationalBasisWeights(ψ, gsIndex)

    return total_time * (log(1 - pd)/log(1-sum(pgs)))
end

