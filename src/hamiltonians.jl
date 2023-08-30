# First, the Hamiltonians correponding to the original QAOA proposal
# We restrict ourselves to the +1 parity sector of the Hilbert space
@doc raw"""
    HxDiagSymmetric(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `HxDiagSymmetric` would be a vector of size ``2^{N-1}``. This construction, only makes sense if the cost/problem 
Hamiltonian ``H_C`` is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HxDiagSymmetric(g::T) where T<: AbstractGraph
    N = nv(g)
    Hx_vec = zeros(ComplexF64, 2^(N-1))
    count = 0
    for j ∈ 0:2^(N-1)-1
        if parity_of_integer(j)==0
            count += 1
            for i ∈ 0:N-1
                Hx_vec[count] += ComplexF64(-2 * (((j >> (i-1)) & 1) ) + 1)
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
function HzzDiagSymmetric(g::T) where T <: AbstractGraph
    N = nv(g)
    matZZ = zeros(ComplexF64, 2^(N-1));
    for edge ∈ edges(g)
        for j ∈ 0:2^(N-1)-1
            matZZ[j+1] += ComplexF64(-2 * (((j >> (edge.src -1)) & 1) ⊻ ((j >> (edge.dst -1)) & 1)) + 1) * getWeight(edge)
        end
    end
    return matZZ
end

function HzzDiagSymmetric(edge::T) where T <: AbstractEdge
    N = nv(g)
    matZZ = zeros(ComplexF64, 2^(N-1));
    for j ∈ 0:2^(N-1)-1
        matZZ[j+1] += ComplexF64(-2 * (((j >> (edge.src -1)) & 1) ⊻ ((j >> (edge.dst -1)) & 1)) + 1) * getWeight(edge)
    end
    return matZZ
end
#####################################################################

function getElementMaxCutHam(x::Int, graph::T) where T <: AbstractGraph
    val = ComplexF64(0)
    for i ∈ edges(graph)
        i_elem = ((x>>(i.src-1))&1)
        j_elem = ((x>>(i.dst-1))&1)
        idx = i_elem ⊻ j_elem
        val += ((-1)^idx)*getWeight(i)
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
function HzzDiag(g::T) where T <: AbstractGraph
    result = ThreadsX.map(x->getElementMaxCutHam(x, g), 0:2^nv(g)-1)
	return result/2
end

function getElementMixingHam(x::Int, graph::T) where T <: AbstractGraph
    val = ComplexF64(0)
    N   = nv(graph)
    for i=1:N
        i_elem = ((x>>(i-1))&1)
        val += (-1)^i_elem
    end
    return val
end

@doc raw"""
    HxDiag(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is ``N``, the corresponding Hamiltonianwould be a vector of size ``2^{N-1}``.
"""
function HxDiag(g::T) where T <: AbstractGraph
    result = ThreadsX.map(x->getElementMixingHam(x, g), 0:2^nv(g)-1)
	return ComplexF64.(result)
end

# Let us also define a function that given a dictionary of "interaction" terms (keys)
# and "weights" (values) constructs the corresponding classical Hamiltonian.
function getElementGeneralClassicalHam(x::Int, interaction::Vector{Int64}, weight::Float64)
    elements = map(i->((x>>(i-1))&1), interaction)
    idx      = foldl(⊻, elements)
    val      = ((-1)^idx)*weight
    return val
end

function getElementGeneralClassicalHam(x::Int, interaction_dict::Dict{Vector{Int64}, Float64})
    val = sum(k->getElementGeneralClassicalHam(x, k, interaction_dict[k]), keys(interaction_dict))
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
function generalClassicalHamiltonian(interaction_dict::Dict{Vector{Int64}, Float64})
    n = reduce(vcat, collect(keys(interaction_dict))) |> Set |> length
    return ThreadsX.map(x->getElementGeneralClassicalHam(x, interaction_dict), 0:2^n-1)
end

