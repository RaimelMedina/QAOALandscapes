function plus_state(T::Type{<:Real}, N::Int)
    return convert(Complex{T}, 2.0^(-N/2)) * ones(Complex{T}, 2^N)
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
function getStateProjection(qaoa::QAOA, params, gsIndex::Vector{Int64})
    ψMin  = getQAOAState(qaoa, params)
    ψ = sum(map(x->_onehot(x, 2^qaoa.N)*ψMin[x], gsIndex))[:]

    normState = norm(ψ)
    normalize!(ψ)

    ψ_perp    = ψMin - (ψ' * ψMin)*ψ
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

function getSmallestEigenvalues(qaoa::QAOA)
    @assert typeof(qaoa.hamiltonian) <: Vector 
    min_energy = minimum(qaoa.hamiltonian |> real)
    return min_energy, findall(x->isapprox(real(x), min_energy), qaoa.hamiltonian)
end

function getSmallestEigenvalues(qaoa::QAOA, k::Int; which = :SR)
    typeHam = typeof(qaoa.hamiltonian)
    if typeHam <: Vector
        println("Returning eigenvalues with position of eigenvectors in the computational basis")
        perm = partialsortperm(qaoa.hamiltonian |> real, 1:k)
        return qaoa.hamiltonian[perm] |> real, perm
    else
        vals, vecs, info = KrylovKit.eigsolve(qaoa.hamiltonian, k, which)
        println("A total of num_eigvals = $(info.converged) were found out of $(k) requested")
        return vals, vecs
    end
end

function timeToSolution(qaoa::QAOA, Γ::AbstractVector{T}; pd=0.99) where T <:Real
    @debug "Here we are assuming that the cost Hamiltonian is classical!"
    p = length(Γ) ÷ 2
    
    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]

    total_time = sum(abs.(γ) + abs.(β))
    _, gsIndex = getSmallestEigenvalues(qaoa)
    ψ = getQAOAState(qaoa, Γ)

    pgs = computationalBasisWeights(ψ, gsIndex)

    return total_time * (log(1.0 - pd)/log(1-sum(pgs)))
end