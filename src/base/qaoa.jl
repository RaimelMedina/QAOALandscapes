@doc raw"""
    QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    QAOA(g::T1, ham::OperatorType{T2}; applySymmetries=true) where {T1 <: AbstractGraph, T2}

Constructors for the `QAOA` object.
"""
struct QAOA{T1 <: AbstractGraph, T2}
    N::Int
    graph::T1
    HB::AbstractVector{T2}
    HC::AbstractVector{T2}
    hamiltonian::OperatorType{T2}
    parity_symmetry::Bool
end

function QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    N = nv(g)
    if applySymmetries==false
        h = 2.0*HzzDiag(g)
        T2 = eltype(h)
        QAOA{T, eltype(h)}(N, g, HxDiag(g), h, h, false)
    else
        h = HzzDiagSymmetric(g)
        T2 = eltype(h)
        QAOA{T, eltype(h)}(N-1, g, HxDiagSymmetric(g), h, h, true) 
    end
end

function QAOA(g::T1, ham::OperatorType{T2}; applySymmetries=true) where {T1 <: AbstractGraph, T2}
    N = nv(g)
    if applySymmetries==false
        QAOA{T1, T2}(N, g, T2.(HxDiag(g)), T2.(HzzDiag(g)), ham, false)
    else
        QAOA{T1, T2}(N-1, g, T2.(HxDiagSymmetric(g)), T2.(HzzDiagSymmetric(g)), ham, true) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N)."
    if qaoa.parity_symmetry
        str2 = "
    Z₂ parity symmetry"
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
function getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T <: Real
    p = length(Γ) ÷ 2
    ψ = plus_state(T, q.N)
    
    @inbounds @simd for i ∈ eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ)
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

function energyVariance(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    h_mean_squared = q(Γ)^2
    ψ = getQAOAState(q, Γ)
    
    typeHam = typeof(q.hamiltonian)
    
    if typeHam <: Vector
        h_squared_mean = dot(ψ, ( q.hamiltonian .^2 ) .* ψ) |> real
    else
        h_squared_mean = dot(ψ, (q.hamiltonian^2), ψ) |> real
    end
    return h_squared_mean - h_mean_squared
end

