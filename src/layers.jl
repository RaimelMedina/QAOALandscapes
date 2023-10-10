function Hx_ψ!(qaoa::QAOA, psi::Vector{Complex{T}}) where T
    N = length(psi)
    num_qubits = Int(log2(N))
    @assert qaoa.N == num_qubits
    
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

function X(qaoa::QAOA, ψ::Vector{Complex{T}}, i::Int) where T
    if qaoa.parity_symmetry
        @assert i ≤ qaoa.N + 1
    else
        @assert i ≤ qaoa.N
    end

    if qaoa.parity_symmetry && i==qaoa.N+1
        return reverse(ψ)
    else
        psi = copy(ψ)
        dim = length(ψ)
        mask = 1 << (i - 1)
        for index in 0:(dim-1)
            if (index & mask) == 0
                psi[index + 1] = psi[index + 1 + mask]
            else
                psi[index + 1] = psi[index + 1 - mask]
            end
        end
        return psi
    end
end

function Hzz_ψ!(qaoa::QAOA, psi::Vector{Complex{T}}) where T<:Real
    if isa(qaoa.hamiltonian, Vector)
        psi .= qaoa.hamiltonian .* psi
    else
        psi .= (qaoa.hamiltonian * psi)
    end
end


function applyExpX!(psi::Vector{Complex{T}}, k::Int, cos_a::T, sin_a::T) where T<:Real
    N = length(psi)
    bitmask = 1 << (k-1)
    for index in 0:(N-1)
        # Check if the k-th bit is unset
        if index & bitmask==0
            psi[index + 1], psi[index + 1 + bitmask] = cos_a * psi[index + 1] - im * sin_a * psi[index + 1 + bitmask], 
                                                              cos_a * psi[index + 1 + bitmask] - im * sin_a * psi[index + 1]
        end
    end
end

function applyExpHB!(psi::Vector{Complex{T}}, β::T; parity_symmetry=false) where T<:Real
    cβ = cos(β)
    sβ = sin(β)
    
    N = length(psi)
    num_qubits = Int(log2(N))
    
    for i ∈ 1:num_qubits
        applyExpX!(psi, i, cβ, sβ)
    end
    if parity_symmetry
        #psi .= (cos(β) .* psi) .- (im * sin(β)) .* reverse(psi)
        for i ∈ 1:N÷2
            psi[i], psi[N-i+1] = cβ * psi[i] - im * sβ * psi[N-i+1],
                                 cβ * psi[N-i+1] - im * sβ * psi[i]
        end
    end
    return nothing
end

function applyExpHC!(q::QAOA, γ::T, ψ0::Vector{Complex{T}}) where T<:Real
    ψ0 .= exp.(-im * (γ .* q.HC)) .* ψ0
end

function applyQAOALayer!(q::QAOA, Γ::AbstractVector{T}, index::Int, ψ0::Vector{Complex{T}}) where T<:Real
    if isodd(index)
        applyExpHC!(q, Γ[index], ψ0)
    else
        applyExpHB!(ψ0, Γ[index]; parity_symmetry = q.parity_symmetry)
    end
end

function applyQAOALayerAdjoint!(q::QAOA, Γ::AbstractVector{T}, index::Int, ψ0::Vector{Complex{T}}) where T<:Real
    if isodd(index)
        applyExpHC!(q, -Γ[index], ψ0) 
    else
        applyExpHB!(ψ0, -Γ[index]; parity_symmetry = q.parity_symmetry)
    end
end

function applyQAOALayerDerivative!(qaoa::QAOA, params::AbstractVector{T}, pos::Int, state::Vector{Complex{T}}) where T<: Real
    if isodd(pos)
        # γ-type parameter
        applyExpHC!(qaoa, params[pos], state)
        Hzz_ψ!(qaoa, state)
        state .*= -1.0*im
    else
        # β-type parameter
        applyExpHB!(state, params[pos]; parity_symmetry=qaoa.parity_symmetry)
        Hx_ψ!(qaoa, state)
        state .*= -1.0*im
        # β-type parameter
        # QAOALandscapes.fwht!(state, qaoa.N)
        # state .= exp.(-im * params[pos] * qaoa.HB) .* state
        # state .= (-im .* qaoa.HB) .* state
        # QAOALandscapes.ifwht!(state, qaoa.N)
    end
end