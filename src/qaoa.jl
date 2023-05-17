@doc raw"""
    QAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))

Constructor for the `QAOA` object.
"""
struct QAOA{T1 <: AbstractGraph, T2}
    N::Int
    graph::T1
    HB::AbstractVector{T2}
    HC::AbstractVector{T2}
    hamiltonian::OperatorType{T2}
end

function QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    if applySymmetries==false
        h = HzzDiag(g)
        QAOA{T, eltype(h)}(nv(g), g, HxDiag(g), h, h)
    else
        h = HzzDiagSymmetric(g)
        QAOA{T, eltype(h)}(nv(g)-1, g, HxDiagSymmetric(g), h, h) 
    end
end

function QAOA(g::T1, ham::OperatorType{T2}; applySymmetries=true) where {T1 <: AbstractGraph, T2}
    if applySymmetries==false
        QAOA{T1, T2}(nv(g), g, T2(HxDiag(g)), T2(HzzDiag(g)), ham)
    else
        QAOA{T1, T2}(nv(g)-1, g, T2(HxDiagSymmetric(g)), T2(HzzDiagSymmetric(g)), ham) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N), and
    graph = $(qaoa.graph)"
    print(io,str)
end

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
    typeHam = typeof(q.hamiltonian)
    if typeHam <: Vector
        return real(ψ' * (q.hamiltonian .* ψ))
    else
        return real(dot(ψ, q.hamiltonian, ψ))
    end
end

@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. At the moment, we do it by using the [`FiniteDiff.jl`](https://github.com/JuliaDiff/FiniteDiff.jl)
"""
function hessianCostFunction(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    matHessian = ForwardDiff.hessian(q, Γ)
    return matHessian
end