@doc raw"""
    QAOA(g::T; applySymmetries=true) where T <: AbstractGraph

Constructors for the `QAOA` object.
"""
struct QAOA{K<:AbstractProblem, T<:AbstractVector, M <: AbstractMixer}
    N::Int
    problem::K
    HC::T
    mixer::M
    initial_state::T
end

function QAOA(cp::ClassicalProblem{R}) where R<:Real
    mixer = XMixer(cp.n)
    ham = hamiltonian(cp)

    T = typeof(ham)
    M = typeof(mixer)
    K = typeof(cp)
    
    if z2SymmetricQ(cp)
        ψ0 = fill(Complex{R}(1/sqrt(2^(cp.n-1))), 2^(cp.n-1))
    else
        ψ0 = fill(Complex{R}(1/sqrt(2^(cp.n))), 2^(cp.n))
    end
    return QAOA{K, T, M}(cp.n, cp, ham, mixer, ψ0)
end
function QAOA(cp::ClassicalProblem{R}, ham::Vector{Complex{R}}, mixer::AbstractMixer) where R<:Real
    T = typeof(ham)
    M = typeof(mixer)
    K = typeof(cp)
    
    if z2SymmetricQ(cp)
        ψ0 = fill(Complex{R}(1/sqrt(2^(cp.n-1))), 2^(cp.n-1))
    else
        ψ0 = fill(Complex{R}(1/sqrt(2^(cp.n))), 2^(cp.n))
    end
    return QAOA{K, T, M}(cp.n, cp, ham, mixer, ψ0)
end

function QAOA(cp::ClassicalProblem{R}, ham::MtlVector{Complex{R}}, mixer::AbstractMixer) where R<:Real
    T = typeof(ham)
    M = typeof(mixer)
    K = typeof(cp)
    
    if z2SymmetricQ(cp)
        ψ0 = Metal.fill(Complex{R}(1/sqrt(2^(cp.n-1))), 2^(cp.n-1))
    else
        ψ0 = Metal.fill(Complex{R}(1/sqrt(2^(cp.n))), 2^(cp.n))
    end
    return QAOA{K, T, M}(cp.n, cp, ham, mixer, ψ0)
end

function Base.show(io::IO, qaoa::QAOA{P, H, M}) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer}
    storage_str = (H <: MtlVector) ? "Metal-GPU" : "CPU"
    str0 = "QAOA object on $(qaoa.N) qubits with mixer type `$(M)`. "
    str1 = "Running on the: -" * storage_str * "- backend."
    print(io, str0 * str1)
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

function getQAOAState(q::QAOA{P, H, M}, Γ::Vector{R}) where {P, H, M, R}
    ψ = copy(q.initial_state)
    for i in eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ)
    end
    return ψ
end

function getQAOAState(q::QAOA{P, H, M}, Γ::Vector{T}, ψ0::H) where {P, H, M, T}
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

function (q::QAOA{P, H, M})(Γ::Vector{R}) where {P, H, M, R}
    ψ = getQAOAState(q, Γ)
    res = real(dot(ψ, q.HC .* ψ)) 
    return res
end

# function (q::QAOA{T1, T2, T3})(Γ::Vector{T}) where {T1 <: AbstractGraph, T2 <: Real, T3 <: AbstractBackend, T}
#     ψ::AbstractVector{Complex{T2}} = getQAOAState(q, Γ)
#     res = real(dot(ψ, q.HC .* ψ)) 
#     return res
# end

# function (q::QAOA{T1, T2, T3})(Γ::Vector{T2}, Etarget::T2) where {T1 <: AbstractGraph, T2 <: Real, T3 <: AbstractBackend}
#     ψ::AbstractVector{Complex{T2}} = getQAOAState(q, Γ)
#     res::T2 = real(dot(ψ, ((q.HC .- Etarget) .^2) .* ψ)) 
#     return res
# end

# function (q::QAOA)(Γ, Etarget)
#     ψ::AbstractVector{Complex{eltype(Γ)}} = getQAOAState(q, Γ) 
#     res = dot(ψ, ((q.HC .- Etarget) .^2) .* ψ)
#     return res |> real
# end

function energyVariance(q::QAOA{T1, T2, T3}, Γ::Vector{T2}) where {T1 <: AbstractGraph, T2 <: Real, T3 <: AbstractBackend}
    h_mean_squared = q(Γ)^2
    ψ = getQAOAState(q, Γ)
    h_squared_mean = dot(ψ, (q.HC .^2) .* ψ) |> real
    return h_squared_mean - h_mean_squared
end

