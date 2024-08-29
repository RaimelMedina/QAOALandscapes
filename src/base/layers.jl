function applyExpX!(psi::Vector{T}, k::Int, cos_a::R, sin_a::R) where {T, R}
    dim = length(psi)
    bitmask = 1 << (k-1)
    @inbounds for index in 0:(dim-1)
        if index & bitmask == 0
            i1 = index + 1
            i2 = index + 1 + bitmask

            # Create local copies to avoid race conditions
            val1 = psi[i1]
            val2 = psi[i2]

            psi[i1] = cos_a * val1 - im * sin_a * val2
            psi[i2] = cos_a * val2 - im * sin_a * val1
        end
    end
    return nothing
end

function applyExpLayer!(mixer::XMixer, psi::Vector{T}, β::R) where {T, R}
    cβ = cos(β)
    sβ = sin(β)
    
    dim = length(psi)
    N = dim |> log2 |> Int
    
    for i ∈ 1:N
        applyExpX!(psi, i, cβ, sβ)
    end
    if N+1 == mixer.N # Z2 symmetric case
        for i ∈ 1:dim÷2
            psi[i], psi[dim-i+1] = cβ * psi[i] - im * sβ * psi[dim-i+1],
                                 cβ * psi[dim-i+1] - im * sβ * psi[i]
        end
    end
    return nothing
end

function applyExpLayer!(hc::Vector{T}, ψ::Vector{K}, γ::R) where {T, K, R}
    for i in eachindex(hc)
        ψ[i] *= exp(-im * γ * hc[i])
    end
    return nothing
end

function applyQAOALayer!(q::QAOA{P, H, M}, elem::T, index::Int, ψ0::Vector{R}) where {P, H, M, T, R}
    if isodd(index)
        applyExpLayer!(q.HC, ψ0, elem)
    else
        applyExpLayer!(q.mixer, ψ0, elem)
    end
    return nothing
end

function applyQAOALayerDerivative!(qaoa::QAOA{P, H, M}, elem::T, pos::Int, state::Vector{R}) where {P, H, M, T, R}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hc_ψ!(qaoa.HC, state)
    else
        qaoa.mixer(state)
    end
    for i in eachindex(state)
        state[i] *= -im
    end
    return nothing
end

function applyQAOALayerDerivative!(qaoa::QAOA{P, H, M}, elem::T, pos::Int, state::Vector{R}, result::Vector{R}) where {P, H, M, T, R}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hc_ψ!(qaoa.HC, state)
    else
        qaoa.mixer(state, result)
    end
    for i in eachindex(state)
        state[i] *= -im
    end
    return nothing
end