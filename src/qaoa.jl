@doc raw"""
    QAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))

Constructor for the `QAOA` object.
"""
mutable struct QAOA{T1 <: AbstractGraph, T2}
    N::Int
    graph::T1
    HB::AbstractVector{T2}
    HC::AbstractVector{T2}
    hamiltonian::OperatorType{T2}
    state::AbstractVector{T2}
    parity_symmetry::Bool
end

function QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    N = nv(g)
    if applySymmetries==false
        h = 2.0*HzzDiag(g)
        T2 = eltype(h)
        ψ0 = fill( T2(2.0^(-N/2)), 2^N)

        QAOA{T, eltype(h)}(N, g, HxDiag(g), h, h, ψ0, false)
    else
        h = HzzDiagSymmetric(g)
        T2 = eltype(h)
        ψ0 = fill( T2(2.0^(-(N-1)/2)), 2^(N-1))
        QAOA{T, eltype(h)}(N-1, g, HxDiagSymmetric(g), h, h, ψ0, true) 
    end
end

function QAOA(g::T1, ham::OperatorType{T2}; applySymmetries=true) where {T1 <: AbstractGraph, T2}
    N = nv(g)
    if applySymmetries==false
        ψ0 = fill( T2(2.0^(-N/2)), 2^N)
        QAOA{T1, T2}(N, g, T2.(HxDiag(g)), T2.(HzzDiag(g)), ham, false)
    else
        ψ0 = fill( T2(2.0^(-(N-1)/2)), 2^(N-1))
        QAOA{T1, T2}(N-1, g, T2.(HxDiagSymmetric(g)), T2.(HzzDiagSymmetric(g)), ham, true) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N),
    "
    if qaoa.parity_symmetry
        str2 = "Z₂ parity symmetry"
        print(io, str * str2)
    else
        print(io, str)
    end
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
function getQAOAState!(q::QAOA, Γ::AbstractVector{T}; use_fwht = false) where T <: Real
    p = length(Γ) ÷ 2
    q.state .= fill( Complex{T}(2.0^(-q.N/2)), 2^q.N)
    if use_fwht
        γ = @view Γ[1:2:2p]
        β = @view Γ[2:2:2p]
        @inbounds @simd for i ∈ 1:p
            q.state .= exp.(-im * (γ[i] .* q.HC)) .* q.state
            fwht!(q.state, q.N)              # Fast Hadamard transformation
            q.state .= exp.(-im * (β[i] .* q.HB)) .* q.state
            ifwht!(q.state, q.N)             # inverse Fast Hadamard transformation
        end
    else
        @inbounds @simd for i ∈ eachindex(Γ)
            applyQAOALayer!(q, Γ, i)
        end
    end
    return nothing
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
    getQAOAState!(q, Γ)
    typeHam = typeof(q.hamiltonian)
    if typeHam <: Vector
        return real(q.state' * (q.hamiltonian .* q.state))
    else
        return real(dot(q.state, q.hamiltonian, q.state))
    end
end

@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. Currently, we use the [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) package
"""
function hessianCostFunction(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    matHessian = ForwardDiff.hessian(q, Γ)
    return matHessian
end