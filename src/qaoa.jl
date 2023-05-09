@doc raw"""
    QAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))

Constructor for the `QAOA` object.
"""
struct QAOA{T1 <: AbstractGraph}
    N::Int
    graph::T1
    HB::AbstractVector{Float64}
    HC::AbstractVector{Float64}
end

function QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    if applySymmetries==false
        QAOA{T}(nv(g), g, HxDiag(g), HzzDiag(g))
    else
        QAOA{T}(nv(g)-1, g, HxDiagSymmetric(g), HzzDiagSymmetric(g)) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N), and
    graph = $(qaoa.graph)"
    print(io,str)
end

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
    val = 0.
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
    val = 0.
    N   = nv(graph)
    for i=1:N
        i_elem = ((x>>(i-1))&1)
        val += (-1)^i_elem
    end
    return val
end

@doc raw"""
    HzzDiag(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is $N$, the corresponding Hamiltonianwould be a vector of size ``2^{N-1}``.
"""
function HxDiag(g::T) where T <: AbstractGraph
    result = ThreadsX.map(x->getElementMixingHam(x, g), 0:2^nv(g)-1)
	return result
end
#####################################################################

@doc raw"""
    getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T <: Real

Construct the QAOA state. More specifically, it returns the state:

```math
    |\Gamma^p \rangle = U(\Gamma^p) |+\rangle
```
with
```math
    U(\Gamma^p) = \prod_{l=1}^p e^{-i H_{B} \beta_{2l}} e^{-i H_{C} \gamma_{2l-1}}
```
and ``H_B, H_C`` corresponding to the mixing and cost Hamiltonian correspondingly.
"""
function getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T <: Real
    p = length(Γ) ÷ 2
    
    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]

    # First set the state vector to |+⟩
    ψ = 2^(-q.N/2)*ones(Complex{T}, 2^q.N)

    @inbounds @simd for i ∈ 1:p
        ψ .= exp.(-im * (γ[i] .* q.HC)) .* ψ
        fwht!(ψ, q.N)              # Fast Hadamard transformation
        ψ .= exp.(-im * (β[i] .* q.HB)) .* ψ
        ifwht!(ψ, q.N)             # inverse Fast Hadamard transformation
    end
    return ψ
end

function getQAOAState(q::QAOA, Γ::AbstractVector{T}, ψ0::AbstractVector{Complex{T}}) where T <: Real
    p = length(Γ) ÷ 2
    
    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]
    
    ψ = copy(ψ0)
    @inbounds @simd for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        fwht!(ψ, q.N)              # Fast Hadamard transformation
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ifwht!(ψ, q.N)             # inverse Fast Hadamard transformation
    end
    return ψ
end

@doc raw"""
    (q::QAOA)(Γ::AbstractVector{T}) where T <: Real

Computes the expectation value of the cost function ``H_C`` in the ``|\Gamma^p \rangle`` state. 
More specifically, it returns the following real number:

```math
    E(\Gamma^p) = \langle \Gamma^p |H_C|\Gamma^p \rangle
```
"""
function (q::QAOA)(Γ::AbstractVector{T}) where T <: Real
    ψ = getQAOAState(q, Γ)
    return real(ψ' * (q.HC .* ψ))
end

@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. At the moment, we do it by using the [`FiniteDiff.jl`](https://github.com/JuliaDiff/FiniteDiff.jl)
"""
function hessianCostFunction(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    matHessian = ForwardDiff.hessian(q, Γ)
    return matHessian
end

@doc raw"""
    getStateJacobian(q::QAOA, θ::T) where T <: AbstractVector

Returns the jacobian ``\nabla |\psi\rangle \in M(\mathbb{C}, 2^N \times 2p)``, where ``N`` corresponds to the total 
number of qubits and ``2p`` is the number of paramaters in ``|\psi(\theta)\rangle``. 
"""
function getStateJacobian(q::QAOA, θ::T) where T <: AbstractVector
    f(x) = getQAOAState(q, x)
    matJacobian = FiniteDiff.finite_difference_jacobian(f, ComplexF64.(θ))
    return matJacobian
end


@doc raw"""
    quantumFisherInfoMatrix(q::QAOA, θ::Vector{Float64})

Constructs the Quantum Fisher Information matrix, defined as follows

``
\mathcal{F}_{ij} = 4 \mathop{\rm{Re}}[\langle \partial_i \psi| \partial_j \psi\rangle - \langle \partial_i \psi| \psi\rangle \langle \psi|\partial_j \psi\rangle ]
``
"""
function quantumFisherInfoMatrix(q::QAOA, θ::T) where T <: AbstractVector
    ∇ψ = getStateJacobian(q, θ)
    ψ  = getQAOAState(q, θ)
    w  = ψ' * ∇ψ

    fisherMat = ∇ψ' * ∇ψ - kron(w', w)
    
    return 4.0*real(fisherMat)
end