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

function kernelExpX!(psi::MtlVector{T}, bitmask::Int, cos_a::K, sin_a::K) where {T, K}
    index = thread_position_in_grid_1d() - 1
    if index & bitmask == 0
        i1 = index + 1
        i2 = index + 1 + bitmask

        # Create local copies to avoid race conditions
        val1 = psi[i1]
        val2 = psi[i2]

        psi[i1] = cos_a * val1 - im * sin_a * val2
        psi[i2] = cos_a * val2 - im * sin_a * val1
    end
    return
end

function kernelExpXParity!(psi::MtlVector{T}, dim::Int, cβ::K, sβ::K) where {T, K}
    i = thread_position_in_grid_1d()
    val1 = psi[i]
    val2 = psi[dim-i+1]

    psi[i]       = cβ * val1 - im * sβ * val2
    psi[dim-i+1] = cβ * val2 - im * sβ * val1
    
    return
end

function applyExpX!(psi::MtlVector{T}, k::Int, cos_a::K, sin_a::K) where {T,K}
    dim = length(psi)
    bitmask = 1 << (k-1)
    num_groups = dim ÷ MAX_THREADS
    @metal threads=MAX_THREADS groups=num_groups kernelExpX!(psi, bitmask, cos_a, sin_a)
    return nothing
end

function applyExpLayer!(mixer::XMixer, psi::MtlVector{T}, β::R) where {T, R}
    cβ = cos(β)
    sβ = sin(β)
    
    dim = length(psi)
    N = Int(log2(dim))
    
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS

    # Loop over spins
    for i ∈ 1:N
        applyExpX!(psi, i, cβ, sβ)
    end
    # check if there is parity symmetry
    if N+1 == mixer.N # Z2 symmetric case
        @metal threads=MAX_THREADS groups=num_groups_parity kernelExpXParity!(psi, dim, cβ, sβ)
    end
    return nothing
end


function applyExpLayer!(hc::Vector{T}, ψ::Vector{K}, γ::R) where {T, K, R}
    @inbounds for i in eachindex(hc)
        ψ[i] *= exp(-im * γ * hc[i])
    end
    return nothing
end

function kernelExpHC!(hc::MtlVector{T}, ψ::MtlVector{K}, γ::R) where {T, K, R}
    i = thread_position_in_grid_1d()
    ψ[i] *= exp(-im * γ * hc[i])
    return
end

function applyExpLayer!(hc::MtlVector{T}, ψ::MtlVector{K}, γ::R) where {T, K, R}
    dim = length(ψ)
    num_groups = dim ÷ MAX_THREADS
    @metal threads=MAX_THREADS groups=num_groups kernelExpHC!(hc, ψ, γ)
    return nothing
end

function applyQAOALayer!(q::QAOA{P, H, M}, elem::T, index::Int, ψ0::AbstractVector{R}) where {P, H, M, T, R}
    if isodd(index)
        applyExpLayer!(q.HC, ψ0, elem)
    else
        applyExpLayer!(q.mixer, ψ0, elem)
    end
    return nothing
end
# function applyExpX!(psi::Vector{Complex{T}}, k::Int, cos_a::T, sin_a::T) where T<:Real
#     N = length(psi)
#     bitmask = 1 << (k-1)
#     for index in 0:(N-1)
#         # Check if the k-th bit is unset
#         if index & bitmask==0
#             psi[index + 1], psi[index + 1 + bitmask] = cos_a * psi[index + 1] - im * sin_a * psi[index + 1 + bitmask], 
#                                                               cos_a * psi[index + 1 + bitmask] - im * sin_a * psi[index + 1]
#         end
#     end
#     return nothing
# end

# function applyExpX!(psi::Vector{Complex{T}}, k::Int, cos_a::T, sin_a::T) where T<:Real
#     N = length(psi)
#     bitmask = 1 << (k-1)

#     for index in 0:(N-1)
#         if index & bitmask == 0
#             i1 = index + 1
#             i2 = index + 1 + bitmask

#             # Create local copies to avoid race conditions
#             val1 = psi[i1]
#             val2 = psi[i2]

#             psi[i1] = cos_a * val1 - im * sin_a * val2
#             psi[i2] = cos_a * val2 - im * sin_a * val1

#             #TODO 
#             # changed -im → +im
#             # psi[i1] = cos_a * val1 + im * sin_a * val2
#             # psi[i2] = cos_a * val2 + im * sin_a * val1
#         end
#     end

#     return nothing
# end



# function applyExpX!(psi, k::Int, cos_a, sin_a)
#     N = length(psi)
#     bitmask = 1 << (k-1)
    
#     @inbounds for index in 0:(N-1)
#         if index & bitmask == 0
#             i1 = index + 1
#             i2 = index + 1 + bitmask

#             # Create local copies to avoid race conditions
#             val1 = psi[i1]
#             val2 = psi[i2]

#             psi[i1] = cos_a * val1 - im * sin_a * val2
#             psi[i2] = cos_a * val2 - im * sin_a * val1
#         end
#     end

#     return nothing
# end

# function applyExpHB!(psi::Vector{Complex{T}}, β::T; parity_symmetry=false) where T<:Real
#     cβ = cos(β)
#     sβ = sin(β)
    
#     N = length(psi)
#     num_qubits = Int(log2(N))
    
#     for i ∈ 1:num_qubits
#         applyExpX!(psi, i, cβ, sβ)
#     end
#     if parity_symmetry
#         #psi .= (cos(β) .* psi) .- (im * sin(β)) .* reverse(psi)
#         for i ∈ 1:N÷2
#             psi[i], psi[N-i+1] = cβ * psi[i] - im * sβ * psi[N-i+1],
#                                  cβ * psi[N-i+1] - im * sβ * psi[i]

#             #TODO
#             # psi[i], psi[N-i+1] = cβ * psi[i] + im * sβ * psi[N-i+1],
#             #                    cβ * psi[N-i+1] + im * sβ * psi[i]
#         end
#     end
#     return nothing
# end

# function applyExpHB!(psi, β; parity_symmetry=false)
#     cβ = cos(β)
#     sβ = sin(β)
    
#     N = length(psi)
#     num_qubits = Int(log2(N))
    
#     for i ∈ 1:num_qubits
#         applyExpX!(psi, i, cβ, sβ)
#     end
#     if parity_symmetry
#         #psi .= (cos(β) .* psi) .- (im * sin(β)) .* reverse(psi)
#         for i ∈ 1:N÷2
#             psi[i], psi[N-i+1] = cβ * psi[i] - im * sβ * psi[N-i+1],
#                                  cβ * psi[N-i+1] - im * sβ * psi[i]

#             #TODO
#             # psi[i], psi[N-i+1] = cβ * psi[i] + im * sβ * psi[N-i+1],
#             #                    cβ * psi[N-i+1] + im * sβ * psi[i]
#         end
#     end
#     return nothing
# end

# function applyExpHC!(hc::Vector{Complex{T2}}, γ::T2, ψ0::Vector{Complex{T2}}) where {T2<:Real}
#     #ψ0 .*= exp.(-im * γ * hc)
#     for i in eachindex(hc, ψ0)
#         ψ0[i] *= exp(-im * γ * hc[i])
#     end
#     return nothing
# end

# function applyExpHC!(hc, γ, ψ0)
#     #ψ0 .*= exp.(-im * γ * hc)
#     for i in eachindex(hc)
#         ψ0[i] *= exp(-im * γ * hc[i])
#     end
#     return nothing
# end

# function applyQAOALayer!(q::QAOA{T1, T2}, elem::T2, index::Int, ψ0::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
#     if isodd(index)
#         applyExpHC!(q.HC, elem, ψ0)
#     else
#         applyExpHB!(ψ0, elem; parity_symmetry = q.parity_symmetry)
#     end
#     return nothing
# end

# function applyQAOALayer!(q::QAOA{T1, T2}, elem, index::Int, ψ0) where {T1<:AbstractGraph, T2<:Real}
#     if isodd(index)
#         applyExpHC!(q.HC, elem, ψ0)
#     else
#         applyExpHB!(ψ0, elem; parity_symmetry = q.parity_symmetry)
#     end
#     return nothing
# end

function applyQAOALayerDerivative!(qaoa::QAOA{P, H, M}, elem::T, pos::Int, state::AbstractVector{R}) where {P, H, M, T, R}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hc_ψ!(qaoa, state)
    else
        qaoa.mixer(state)
    end
    for i in eachindex(state)
        state[i] *= Complex{T2}(-im)
    end
    return nothing
end

# function applyQAOALayerDerivative!(qaoa::QAOA{T1, T2}, elem::T2, pos::Int, state::AbstractVector{Complex{T2}}, result::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
#     applyQAOALayer!(qaoa, elem, pos, state)
#     if isodd(pos)
#         Hzz_ψ!(qaoa, state)
#     else
#         Hx_ψ!(qaoa, state, result)
#     end
#     for i in eachindex(state)
#         state[i] *= Complex{T2}(-im)
#     end
#     return nothing
# end