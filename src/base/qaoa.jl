@doc raw"""
    QAOA(g::T; applySymmetries=true) where T <: AbstractGraph

Constructors for the `QAOA` object.
"""
struct QAOA{T1 <: AbstractGraph, T2 <: Real, T3<:AbstractBackend}
    N::Int
    graph::T1
    HB::AbstractVector{Complex{T2}}
    HC::AbstractVector{Complex{T2}}
    initial_state::AbstractVector{Complex{T2}}
    parity_symmetry::Bool
end

function QAOA(T2::Type{<:Real}, g::T1; applySymmetries=true) where T1 <: AbstractGraph
    N = nv(g)
    if applySymmetries==false
        h = 2*HzzDiag(Complex{T2}, g)
        QAOA{T1, T2, CPUBackend}(N, g, HxDiag(Complex{T2}, g), h, plus_state(CPUBackend, T2, N), false)
    else
        h = HzzDiagSymmetric(Complex{T2}, g)
        QAOA{T1, T2, CPUBackend}(N-1, g, HxDiagSymmetric(Complex{T2}, g), h, plus_state(CPUBackend, T2, N-1), true) 
    end
end

function QAOA(T3::Type{METALBackend}, ::Type{<:Float32}, g::T1; applySymmetries=true) where {T1 <: AbstractGraph}
    N = nv(g)
    T2 = Float32
    if applySymmetries==false
        h = MtlArray(2*HzzDiag(Complex{T2}, g))
        QAOA{T1, T2, METALBackend}(N, g, MtlArray(HxDiag(Complex{T2}, g)), h, plus_state(T3, T2, N), false)
    else
        h = MtlArray(HzzDiagSymmetric(Complex{T2}, g))
        QAOA{T1, T2, METALBackend}(N-1, g, MtlArray(HxDiagSymmetric(Complex{T2}, g)), h, plus_state(T3, T2, N-1), true) 
    end
end

function Base.show(io::IO, qaoa::QAOA{T1, T2, T3}) where {T1 <: AbstractGraph, T2<:Real, T3<:METALBackend}
    str = "QAOA object with:
    running on the Metal backend, 
    number of qubits = $(qaoa.N)."
    if qaoa.parity_symmetry
        str2 = "
    Z₂ parity symmetry"
        print(io, str * str2)
    else
        print(io, str)
    end
end

function Base.show(io::IO, qaoa::QAOA{T1, T2, T3}) where {T1 <: AbstractGraph, T2<:Real, T3<:CPUBackend}
    str = "QAOA object with:
    running on the CPU, 
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
function getQAOAState(q::QAOA{T1, T2, T3}, Γ::Vector{T2}) where {T1 <: AbstractGraph, T2 <: Real, T3<:AbstractBackend}
    ψ::AbstractVector{Complex{T2}} = copy(q.initial_state)
    for i in eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ)
    end
    return ψ
end

function getQAOAState(q::QAOA{T1, T2, T3}, Γ::Vector{T2}, ψ0::AbstractVector{Complex{T2}}) where {T1 <: AbstractGraph, T2 <: Real, T3<:AbstractBackend}
    ψ = copy(ψ0)
    for i in eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ)
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
function (q::QAOA{T1, T2, T3})(Γ::Vector{T2}) where {T1 <: AbstractGraph, T2 <: Real, T3 <: AbstractBackend}
    ψ::AbstractVector{Complex{T2}} = getQAOAState(q, Γ)
    res::T2 = real(dot(ψ, q.HC .* ψ)) 
    return res
end

function energyVariance(q::QAOA{T1, T2, T3}, Γ::Vector{T2}) where {T1 <: AbstractGraph, T2 <: Real, T3 <: AbstractBackend}
    h_mean_squared = q(Γ)^2
    ψ = getQAOAState(q, Γ)
    h_squared_mean = dot(ψ, (q.HC .^2) .* ψ) |> real
    return h_squared_mean - h_mean_squared
end

