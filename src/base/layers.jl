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
    return nothing
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
    return nothing
end

function applyQAOALayer!(q::QAOA, elem::T, index::Int, ψ0::Vector{Complex{T}}) where T<:Real
    if isodd(index)
        applyExpHC!(q, elem, ψ0)
    else
        applyExpHB!(ψ0, elem; parity_symmetry = q.parity_symmetry)
    end
    return nothing
end

# function applyQAOALayerAdjoint!(q::QAOA, Γ::AbstractVector{T}, index::Int, ψ0::Vector{Complex{T}}) where T<:Real
#     if isodd(index)
#         applyExpHC!(q, -Γ[index], ψ0) 
#     else
#         applyExpHB!(ψ0, -Γ[index]; parity_symmetry = q.parity_symmetry)
#     end
# end

function applyQAOALayerDerivative!(qaoa::QAOA, elem::T, pos::Int, state::Vector{Complex{T}}) where T<: Real
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        # γ-type parameter
        #applyExpHC!(qaoa, params[pos], state)
        Hzz_ψ!(qaoa, state)
    else
        # β-type parameter
        #applyExpHB!(state, params[pos]; parity_symmetry=qaoa.parity_symmetry)
        Hx_ψ!(qaoa, state)
        #state .*= -1.0*im
        # β-type parameter
        # QAOALandscapes.fwht!(state, qaoa.N)
        # state .= exp.(-im * params[pos] * qaoa.HB) .* state
        # state .= (-im .* qaoa.HB) .* state
        # QAOALandscapes.ifwht!(state, qaoa.N)
    end
    state .*= -1.0*im
    return nothing
end