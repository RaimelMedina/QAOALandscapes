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

function applyExpX!(psi::Vector{Complex{T}}, k::Int, cos_a::T, sin_a::T) where T<:Real
    N = length(psi)
    bitmask = 1 << (k-1)

    for index in 0:(N-1)
        if index & bitmask == 0
            i1 = index + 1
            i2 = index + 1 + bitmask

            # Create local copies to avoid race conditions
            val1 = psi[i1]
            val2 = psi[i2]

            psi[i1] = cos_a * val1 - im * sin_a * val2
            psi[i2] = cos_a * val2 - im * sin_a * val1

            #TODO 
            # changed -im → +im
            # psi[i1] = cos_a * val1 + im * sin_a * val2
            # psi[i2] = cos_a * val2 + im * sin_a * val1
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

            #TODO
            # psi[i], psi[N-i+1] = cβ * psi[i] + im * sβ * psi[N-i+1],
            #                    cβ * psi[N-i+1] + im * sβ * psi[i]
        end
    end
    return nothing
end

function applyExpHC!(hc::Vector{Complex{T2}}, γ::T2, ψ0::Vector{Complex{T2}}) where {T2<:Real}
    #ψ0 .*= exp.(-im * γ * hc)
    Threads.@threads for i in eachindex(hc, ψ0)
        ψ0[i] *= exp(-im * γ * hc[i])
    end
    return nothing
end

function applyQAOALayer!(q::QAOA{T1, T2}, elem::T2, index::Int, ψ0::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    if isodd(index)
        applyExpHC!(q.HC, elem, ψ0)
    else
        applyExpHB!(ψ0, elem; parity_symmetry = q.parity_symmetry)
    end
    return nothing
end


function applyQAOALayerDerivative!(qaoa::QAOA{T1, T2}, elem::T2, pos::Int, state::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hzz_ψ!(qaoa, state)
    else
        Hx_ψ!(qaoa, state)
    end
    for i in eachindex(state)
        state[i] *= Complex{T2}(-im)
    end
    return nothing
end

function applyQAOALayerDerivative!(qaoa::QAOA{T1, T2}, elem::T2, pos::Int, state::Vector{Complex{T2}}, result::Vector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hzz_ψ!(qaoa, state)
    else
        Hx_ψ!(qaoa, state, result)
    end
    for i in eachindex(state)
        state[i] *= Complex{T2}(-im)
    end
    return nothing
end