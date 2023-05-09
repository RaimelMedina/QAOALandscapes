"""
    getGroundStateProjection(qaoa::QAOA, params, gsIndex::Vector{Int64})

Calculates the projection of the QAOA state onto the ground state (GS) subspace and its orthogonal complement. 
The QAOA state is determined by the given parameters `params`.

# Arguments
* `qaoa::QAOA`: QAOA object.
* `params`: Parameters determining the QAOA state.
* `gsIndex::Vector{Int64}`: Indices of the ground state components.

# Returns
* `normGS`: Norm of the projection of the QAOA state onto the ground state subspace.
* `normGS_perp`: Norm of the projection of the QAOA state onto the orthogonal complement of the ground state subspace.
* `ψGS`: Normalized projection of the QAOA state onto the ground state subspace.
* `ψGS_perp`: Normalized projection of the QAOA state onto the orthogonal complement of the ground state subspace.
"""
function getGroundStateProjection(qaoa::QAOA, params, gsIndex::Vector{Int64})
    ψMin  = getQAOAState(qaoa, params)
    ψGS = sum(map(x->_onehot(x, 2^qaoa.N)*ψMin[x], gsIndex))[:]

    normGS = norm(ψGS)
    normalize!(ψGS)

    ψGS_perp    = ψMin - (ψGS' * ψMin)*ψGS
    normGS_perp = norm(ψGS_perp)
    normalize!(ψGS_perp)

    return normGS, normGS_perp, ψGS, ψGS_perp
end

"""
    getStateEquivClasses(qaoa::QAOA)

Computes the equivalence classes of states based on their energies in the QAOA problem. 
The energies are rounded to a certain number of significant digits (default is 5) to group the states with approximately equal energies.

# Arguments
* `qaoa::QAOA`: QAOA object.
* `sigdigits=5`: Significant digits to which energies are rounded.

# Returns
* `data_states`: A Vector of Vectors, where each inner Vector contains the indices of the states belonging to the same energy equivalence class.
"""
function getStateEquivClasses(qaoa::QAOA; sigdigits=5)
    unique_energies = round.(qaoa.HC, sigdigits=sigdigits) |> unique |> sort
    data_states = Vector{Vector{Int64}}()
    for x ∈ unique_energies
        push!(data_states, findall(s->isapprox(s, x), qaoa.HC))
    end
    return data_states
end

"""
    computationalBasisWeights(ψ, equivClasses)

Computes the computational basis weights for a given state vector `ψ` according to the provided equivalence classes, by summing up the squared magnitudes of the elements with the same equivalence class.

# Arguments
* `ψ`: The state vector.
* `equivClasses`: A Vector of Vectors, where each inner Vector contains the indices of the elements belonging to the same equivalence class.

# Returns
* `basis_weights`: A Vector containing the summed squared magnitudes of `ψ` for each equivalence class.
"""
function computationalBasisWeights(ψ, equivClasses)
    return map(x-> sum(abs2.(getindex(ψ, x))), equivClasses)
end