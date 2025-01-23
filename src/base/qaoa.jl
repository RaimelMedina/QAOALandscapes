@doc raw"""
    QAOA{C<:AbstractQAOACost, E<:ExpectationMethod, K<:AbstractProblem, T<:AbstractVector, M<:AbstractMixer}

Type representing a QAOA (Quantum Approximate Optimization Algorithm) instance.

# Fields
- `N::Int`: Number of qubits
- `problem::K`: The optimization problem instance
- `HC::T`: Cost Hamiltonian
- `mixer::M`: Mixing operator
- `initial_state::T`: Initial state for the QAOA circuit
- `nshots::Union{Int, Nothing}`: Number of measurement shots (only for `E <: SamplingMethod`)

# Type Parameters
- `C`: Type of cost function (Classical or Quantum)
- `E`: Expectation value method (ExactMethod or SamplingMethod)
- `K`: Problem type
- `T`: Vector type for states and Hamiltonians
- `M`: Mixer type
"""
struct QAOA{C<:AbstractQAOACost, E<:ExpectationMethod, K<:AbstractProblem, T<:AbstractVector, M<:AbstractMixer}
    N::Int
    problem::K
    HC::T
    mixer::M
    nshots::Union{Int, Nothing}  # Only used when E <: SamplingMethod
end

"""
    QAOA(cp::ClassicalProblem{R}) where {R<:Real}

Construct a QAOA instance for a classical problem using exact expectation value calculations.

# Arguments
- `cp::ClassicalProblem{R}`: Classical optimization problem instance

# Returns
- `QAOA` instance configured for exact expectation value calculations
"""
function QAOA(cp::ClassicalProblem{R}) where {R<:Real}
    mixer = XMixer(cp.n)
    ham = hamiltonian(cp)
    T = typeof(ham)
    M = typeof(mixer)
    K = typeof(cp)
    return QAOA{ClassicalCost, ExactMethod, K, T, M}(cp.n, cp, ham, mixer, nothing)
end

function QAOA(cp::ClassicalProblem{R}, ham::S) where {R<:Real, S<:AbstractGPUArray{Complex{R}}}
    mixer = XMixer(cp.n)
    K = typeof(cp)
    return QAOA{ClassicalCost, ExactMethod, K, S, typeof(mixer)}(cp.n, cp, ham, mixer, nothing)
end

@doc raw"""
    QAOA{C}(cp::ClassicalProblem{R}, nshots::Int) where {R<:Real, C<:AbstractQAOACost}

Construct a QAOA instance for a given cost type using shot-based measurements.

# Arguments
- `cp::ClassicalProblem{R}`: Classical optimization problem instance
- `nshots::Int`: Number of measurement shots to use for expectation value estimation

# Returns
- `QAOA` instance configured for shot-based measurements
"""
function QAOA(cp::ClassicalProblem{R}, nshots::Int) where {R<:Real}
    mixer = XMixer(cp.n)
    ham = hamiltonian(cp)

    T = typeof(ham)
    M = typeof(mixer)
    K = typeof(cp)
    
    return QAOA{ClassicalCost, SamplingMethod, K, T, M}(cp.n, cp, ham, mixer, nshots)
end

# Show method
function Base.show(io::IO, qaoa::QAOA{C, E, P, H, M}) where {C, E, P, H, M}
    storage_str = (H <: AbstractGPUArray) ? "GPU" : "CPU"
    println(io, "QAOA object on $(qaoa.N) qubits with mixer type `$(M)`")
    println(io, "Running on the " * storage_str * " backend")
    str2 = if E === ExactMethod
        "Using exact expectation values"
    else
        "Using sampling with $(qaoa.nshots) shots"
    end
    println(io, str2)
end

@doc raw"""
    getQAOAState(q::QAOA, Γ::AbstractVector{T}) where {T}

Construct the QAOA state. More specifically, it returns the state:

```math
    |\Gamma^p \rangle = U(\Gamma^p) |+\rangle
```
with
```math
    U(\Gamma^p) = \prod_{l=1}^p e^{-i H_{B} \beta_{2l}} e^{-i H_{C} \gamma_{2l-1}}
```
and ``H_B, H_C`` corresponding to the mixing and cost Hamiltonian respectively.
"""
function getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    dim = length(q.HC)

    ψ::AbstractVector{Complex{T}} = similar(q.HC)
    ψ .= Complex{T}(1/sqrt(dim))
    
    for i in eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ)
    end
    return ψ
end

@doc raw"""
    getQAOAState(q::QAOA, Γ::AbstractVector{T}, ψ0::H) where {H, T}

Construct the QAOA state. The main difference here is that it uses the state ``ψ0`` as initial state instead of $|+\rangle$. That is:

```math
    |\Gamma^p \rangle = U(\Gamma^p) |\psi_0\rangle
```
with
```math
    U(\Gamma^p) = \prod_{l=1}^p e^{-i H_{B} \beta_{2l}} e^{-i H_{C} \gamma_{2l-1}}
```
and ``H_B, H_C`` corresponding to the mixing and cost Hamiltonian respectively.
"""
function getQAOAState(q::QAOA, Γ::AbstractVector{T}, ψ0::H) where {H, T<:Real}
    dim = length(q.HC)
    @assert dim == length(ψ0)

    ψ0 .= Complex{T}(1/sqrt(dim))
    
    for i in eachindex(Γ)
        applyQAOALayer!(q, Γ[i], i, ψ0)
    end
    return ψ0
end

@doc raw"""
    (q::QAOA{C, ExactMethod, P, H, M})(Γ::AbstractVector{R}) where {C<:ClassicalCost, P, H, M, R}

Compute the exact expectation value of the classical cost Hamiltonian Hᶜ for the QAOA state.

# Arguments
- `Γ::AbstractVector{R}`: QAOA parameters [γ₁, β₁, γ₂, β₂, ...]

# Returns
The expectation value E(Γ) = ⟨Γᵖ|Hᶜ|Γᵖ⟩
"""
function (q::QAOA{C, ExactMethod, P, H, M})(Γ::AbstractVector{T}) where {C<:ClassicalCost, P, H, M, T<:Real}
    ψ = getQAOAState(q, Γ)
    return real(dot(ψ, q.HC .* ψ))
end

@doc raw"""
    (q::QAOA{C, SamplingMethod, P, H, M})(Γ::AbstractVector{R}) where {C<:ClassicalCost, P, H, M, R}

Estimate the expectation value of the classical cost Hamiltonian Hᶜ using shot-based measurements.

# Arguments
- `Γ::AbstractVector{R}`: QAOA parameters [γ₁, β₁, γ₂, β₂, ...]

# Returns
Shot-based estimate of E(Γ) = ⟨Γᵖ|Hᶜ|Γᵖ⟩ using q.nshots measurements
"""
function (q::QAOA{C, SamplingMethod, P, H, M})(Γ::AbstractVector{R}) where {C<:ClassicalCost, P, H, M, R}
    ψ = getQAOAState(q, Γ)
    probabilities = abs2.(ψ)
    samples = sample(1:length(ψ), Weights(probabilities), q.nshots)
    return mean(real.(q.HC[samples]))
end

@doc raw"""
    (q::QAOA{C, ExactMethod, P, H, M})(Γ::AbstractVector{R}) where {P, H, M, R, C<:QuantumCost}

Compute the exact expectation value for quantum cost functions, including both cost and mixer terms.

# Arguments
- `Γ::AbstractVector{R}`: QAOA parameters [γ₁, β₁, γ₂, β₂, ...]

# Returns
The expectation value E(Γ) = ⟨Γᵖ|Hᶜ|Γᵖ⟩ + ⟨Γᵖ|Hᵦ|Γᵖ⟩ where Hᵦ is the mixing Hamiltonian
"""
function (q::QAOA{C, ExactMethod, P, H, M})(Γ::AbstractVector{R}) where {P, H, M, R, C<:QuantumCost}
    ψ = getQAOAState(q, Γ)
    res_hc = real(dot(ψ, q.HC .* ψ))
    ψhx = copy(ψ)
    q.mixer(ψhx)
    res_hb = real(dot(ψ, ψhx)) 
    return res_hc + res_hb
end

@doc raw"""
    energyVariance(q::QAOA{C, ExactMethod, P, H, M}, Γ::AbstractVector{T}) where {C<:ClassicalCost, P, H, M, T<:Real}
    energyVariance(q::QAOA{C, ExactMethod, P, H, M}, ψ::AbstractVector{Complex{T}}) where {C<:QuantumCost, P, H, M, T<:Real}

Computes the energy variance of the cost Hamiltonian $H_C$ ($H_C + H_B$) in the QAOA state. 
Alternatively, computes the energy variance of the cost Hamiltonian in a given state $\psi$:

```math
    \mathrm{var}_{\Gamma}[H_C] = \langle \Gamma^p |H_C^2|\Gamma^p \rangle-\langle \Gamma^p |H_C|\Gamma^p \rangle^2
```
"""
function energyVariance(q::QAOA{C, ExactMethod, P, H, M}, Γ::AbstractVector{T}) where {C<:ClassicalCost, P, H, M, T<:Real}
    h_mean_squared = q(Γ)^2
    ψ = getQAOAState(q, Γ)
    h_squared_mean = dot(ψ, (q.HC .^2) .* ψ) |> real
    return h_squared_mean - h_mean_squared
end

function energyVariance(q::QAOA{C, ExactMethod, P, H, M}, ψ::AbstractVector{Complex{T}}) where {C<:ClassicalCost, P, H, M, T}
    h_mean_squared = real(dot(ψ, q.HC .* ψ))^2
    h_squared_mean = dot(ψ, (q.HC .^2) .* ψ) |> real
    return h_squared_mean - h_mean_squared
end
