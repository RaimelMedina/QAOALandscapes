# First, the Hamiltonians correponding to the original QAOA proposal
# We restrict ourselves to the +1 parity sector of the Hilbert space
@doc raw"""
    HxDiagSymmetric(T::Type{<:Real}, g::S) where S <: AbstractGraph

Construct the mixing Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `HxDiagSymmetric` would be a vector of size ``2^{N-1}``. This construction, only makes sense if the cost/problem 
Hamiltonian ``H_C`` is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HxDiagSymmetric(T::Type{Complex{R}}, g::T2) where {R<:Real, T2<: AbstractGraph}
    N = nv(g)
    Hx_vec = zeros(T, 2^(N-1))
    count = 0
    for j ∈ 0:2^(N-1)-1
        if parity_of_integer(j)==0
            count += 1
            for i ∈ 0:N-1
                Hx_vec[count] += T(-2 * (((j >> (i-1)) & 1) ) + 1)
            end
            Hx_vec[2^(N-1) - count + 1] = - Hx_vec[count]
        end
    end
    return Hx_vec
end

@doc raw"""
    HzzDiagSymmetric(g::T) where T <: AbstractGraph
    HzzDiagSymmetric(edge::T) where T <: AbstractEdge

Construct the cost Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `HzzDiagSymmetric` would be a vector of size ``2^{N-1}``. This construction, only makes sense if the cost/problem 
Hamiltonian ``H_C`` is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
Similarly, if the input is an `edge` then it returns the corresponding ``ZZ`` operator.
"""
function HzzDiagSymmetric(T::Type{Complex{R}}, g::T2) where {R<:Real, T2 <: AbstractGraph}
    N = nv(g)
    matZZ = zeros(T, 2^(N-1));
    for edge ∈ edges(g)
        for j ∈ 0:2^(N-1)-1
            matZZ[j+1] += T(-2 * (((j >> (edge.src -1)) & 1) ⊻ ((j >> (edge.dst -1)) & 1)) + 1) * getWeight(R, edge)
        end
    end
    return matZZ
end

function HzzDiagSymmetric(T::Type{Complex{R}}, g::T1, edge::T2) where {R<:Real, T1<:AbstractGraph, T2 <: AbstractEdge}
    N = nv(g)
    matZZ = zeros(T, 2^(N-1));
    for j ∈ 0:2^(N-1)-1
        matZZ[j+1] += T(-2 * (((j >> (edge.src -1)) & 1) ⊻ ((j >> (edge.dst -1)) & 1)) + 1) * getWeight(R, edge)
    end
    return matZZ
end
#####################################################################

function getElementMaxCutHam(T::Type{Complex{R}}, x::Int, graph::T2) where {R<:Real, T2 <: AbstractGraph}
    val = T(0)
    for i ∈ edges(graph)
        i_elem = ((x>>(i.src-1))&1)
        j_elem = ((x>>(i.dst-1))&1)
        idx = i_elem ⊻ j_elem
        val += T(((-1)^idx)*getWeight(R, i))
    end
    return val
end

@doc raw"""
    HzzDiag(g::T) where T <: AbstractGraph

Construct the cost Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is ``N``, the corresponding Hamiltonian would be a vector of size ``2^{N-1}``.
This function instead returs a vector of size ``2^N``. 
"""
function HzzDiag(T::Type{Complex{R}}, g::T2) where {R<:Real, T2 <: AbstractGraph}
    result = ThreadsX.map(x->getElementMaxCutHam(T, x, g), 0:2^nv(g)-1)
	return result/2
end

function getElementMixingHam(T::Type{Complex{R}}, x::Int, graph::T2) where {R<:Real, T2 <: AbstractGraph}
    val = T(0)
    N   = nv(graph)
    for i=1:N
        i_elem = ((x>>(i-1))&1)
        val += T((-1)^i_elem)
    end
    return val
end

@doc raw"""
    HxDiag(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is ``N``, the corresponding Hamiltonianwould be a vector of size ``2^{N-1}``.
"""
function HxDiag(T::Type{Complex{R}}, g::T2) where {R<:Real, T2 <: AbstractGraph}
    #result = ThreadsX.map(x->getElementMixingHam(T, x, g), 0:2^nv(g)-1)
    result = ThreadsX.map(x->-getElementMixingHam(T, x, g), 0:2^nv(g)-1)
    # I changed so that HB → -HB which is the correct Hamiltonian
	return result
end

# Let us also define a function that given a dictionary of "interaction" terms (keys)
# and "weights" (values) constructs the corresponding classical Hamiltonian.
function getElementGeneralClassicalHam(T::Type{Complex{R}}, x::Int, interaction::Vector{Int}, weight::T2) where {R<:Real, T2 <: Real}
    elements = map(i->((x>>(i-1))&1), interaction)
    idx      = foldl(⊻, elements)
    val      = T(((-1)^idx)*weight)
    return val
end

function getElementGeneralClassicalHam(T::Type{Complex{R}}, x::Int, interaction_dict::Dict{Vector{Int}, T2}) where {R<:Real, T2 <: Real}
    val = sum(k->getElementGeneralClassicalHam(T, x, k, interaction_dict[k]), keys(interaction_dict))
    return val
end

@doc raw"""
    generalClassicalHamiltonian(interaction_dict::Dict{Vector{Int64}, Float64})

This function computes the classical Hamiltonian for a general ``p``-spin Hamiltonian, that is
```math
    H_Z  = \sum_{i \in S} J_i \prod_{\alpha \in i} \sigma^z_{i_{\alpha}}
```
Above, ``S`` is the set of interaction terms which is passed as an argument in the for of a dictionary with keys
being the spins participating in a given interaction and values given by the weights of such interaction term.

# Arguments
- `interaction_dict`: a dictionary where each key is a vector of integers representing an interaction, and each value is the weight of that interaction.

# Returns
- `hamiltonian::Vector{Float64}`: The general ``p`` spin Hamiltonian.
"""
function generalClassicalHamiltonian(T::Type{Complex{R}}, n::Int, interaction_dict::Dict{Vector{Int}, T2}) where {R<:Real, T2 <: Real}
    return ThreadsX.map(x->getElementGeneralClassicalHam(T, x, interaction_dict), 0:2^n-1)
end


function Hx_ψ!(qaoa::QAOA{T1, T2}, psi::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    N = length(psi)
    num_qubits = Int(log2(N))
    @assert qaoa.N == num_qubits
    
    #FIXME
    # multiplied by -1 the whole state
    # psi .*= Complex{T2}(-1)
    result = copy(psi)

    
    for qubit in 1:num_qubits
        mask = 1 << (qubit - 1)
        for index in 0:(N-1)
            if (index & mask) == 0
                psi[index + 1] += result[index + 1 + mask]
            else
                psi[index + 1] += result[index + 1 - mask]
            end
        end
    end
    if qaoa.parity_symmetry
        for l ∈ 1:N÷2
            psi[l] += result[N-l+1]
            psi[N-l+1] += result[l]
        end
    end
    return nothing
end

function Hx_ψ!(qaoa::QAOA{T1, T2}, psi::Vector{Complex{T2}}, result::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    N = length(psi)
    num_qubits = Int(log2(N))
    @assert qaoa.N == num_qubits
    

    #FIXME
    # multiplied by -1 the whole state
    # psi .*= Complex{T2}(-1)
    result .= psi

    for qubit in 1:num_qubits
        mask = 1 << (qubit - 1)
        for index in 0:(N-1)
            if (index & mask) == 0
                psi[index + 1] += result[index + 1 + mask]
            else
                psi[index + 1] += result[index + 1 - mask]
            end
        end
    end
    if qaoa.parity_symmetry
        for l ∈ 1:N÷2
            psi[l] += result[N-l+1]
            psi[N-l+1] += result[l]
        end
    end
    return nothing
end


function Hzz_ψ!(qaoa::QAOA{T1, T2}, psi::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    Threads.@threads for i in eachindex(qaoa.HC, psi)
        psi[i] *= qaoa.HC[i]
    end
    return nothing
end