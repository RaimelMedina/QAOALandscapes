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
    length(gsIndex) == 1 ? nothing : ArgumentError("Ground state is not unique! This could lead to wrong results")
    ψMin  = getQAOAState(qaoa, params)
    # α = ⟨E₀|Γ⟩
    
    α  = getindex(ψMin, gsIndex[1])
    GS = _onehot(Complex{T}, gsIndex[1], length(qaoa.initial_state))

    # |ψₚ⟩ = |Γ⟩ - α |E₀⟩

    ψp = ψMin - α * GS
    norm_ψp = norm(ψp)
    normalize!(ψp)

    @assert abs2(α) + norm_ψp^2 ≈ T(1)

    return α, norm_ψp, GS, ψp
end

function getStateProjection(qaoa::QAOA{P, H, M}, ψinit::AbstractVector{Complex{T}}, gsIndex::Vector{Int}) where {P, H, M, T<:Real}
    length(gsIndex) == 1 ? nothing : ArgumentError("Ground state is not unique! This could lead to wrong results")
    
    ψMin  = copy(ψinit)
    # α = ⟨E₀|Γ⟩
    
    α  = getindex(ψMin, gsIndex[1])
    GS = _onehot(Complex{T}, gsIndex[1], length(qaoa.initial_state))

    # |ψₚ⟩ = |Γ⟩ - α |E₀⟩

    ψp = ψMin - α * GS
    norm_ψp = norm(ψp)
    normalize!(ψp)

    @assert abs2(α) + norm_ψp^2 ≈ T(1)

    return α, norm_ψp, GS, ψp
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

