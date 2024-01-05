#### First come the definitions needed to construct the QAOA state ####
function kernelExpX!(psi::AbstractVector{Complex{T}}, bitmask::Int, cos_a::T, sin_a::T) where T<:Real
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

function kernelExpXParity!(psi::AbstractVector{Complex{T}}, dim::Int, cβ::T, sβ::T) where T<:Real
    i = thread_position_in_grid_1d()
    val1 = psi[i]
    val2 = psi[dim-i+1]

    psi[i]       = cβ * val1 - im * sβ * val2
    psi[dim-i+1] = cβ * val2 - im * sβ * val1
    
    return
end


function applyExpX!(::Type{METALBackend}, psi::AbstractVector{Complex{T}}, k::Int, cos_a::T, sin_a::T) where T<:Real
    dim = length(psi)
    bitmask = 1 << (k-1)
    num_groups = dim ÷ MAX_THREADS
    @metal threads=MAX_THREADS groups=num_groups kernelExpX!(psi, bitmask, cos_a, sin_a)
    return nothing
end

function applyExpHB!(::Type{METALBackend}, psi::AbstractVector{Complex{T}}, β::T; parity_symmetry=false) where T<:Real
    cβ = cos(β)
    sβ = sin(β)
    
    dim = length(psi)
    N = Int(log2(dim))
    
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS

    # Loop over spins
    for i ∈ 1:N
        applyExpX!(METALBackend, psi, i, cβ, sβ)
    end

    # check if there is parity symmetry
    if parity_symmetry
        @metal threads=MAX_THREADS groups=num_groups_parity kernelExpXParity!(psi, dim, cβ, sβ)
    end
    return nothing
end

function kernelExpHC!(hc::AbstractVector{Complex{T2}}, γ::T2, ψ0::AbstractVector{Complex{T2}}) where {T2<:Real}
    i = thread_position_in_grid_1d()
    ψ0[i] *= exp(-im * γ * hc[i])
    return
end

function applyExpHC!(::Type{METALBackend}, hc::AbstractVector{Complex{T2}}, γ::T2, ψ0::AbstractVector{Complex{T2}}) where {T2<:Real}
    dim = length(ψ0)
    num_groups = dim ÷ MAX_THREADS

    @metal threads=MAX_THREADS groups=num_groups kernelExpHC!(hc, γ, ψ0)
    return nothing
end

function applyQAOALayer!(q::QAOA{T1, T2, T3}, elem::T2, index::Int, ψ0::AbstractVector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}
    if isodd(index)
        applyExpHC!(T3, q.HC, elem, ψ0)
    else
        applyExpHB!(T3, ψ0, elem; parity_symmetry = q.parity_symmetry)
    end
    return nothing
end


#### kernels for HC|ψ⟩ and HB|ψ⟩ ####
function kernelHx!(psi::AbstractVector{Complex{T}}, bitmask::Int, result::AbstractVector{Complex{T}}) where T<:Real
    index = thread_position_in_grid_1d() - 1
    i1 = index + 1
    if index & bitmask == 0
        i2 = index + 1 + bitmask
    else
        i2 = index + 1 - bitmask
    end
    psi[i1] += result[i2]
    return
end

function kernelHxParity!(psi::AbstractVector{Complex{T}}, dim::Int, result::AbstractVector{Complex{T}}) where T<:Real
    i = thread_position_in_grid_1d()

    psi[i]       += result[dim-i+1]
    psi[dim-i+1] += result[i]
    
    return
end

function Hx_ψ!(qaoa::QAOA{T1, T2, T3}, psi::AbstractVector{Complex{T2}}, result::AbstractVector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}
    dim = length(psi)
    N = Int(log2(dim))

    num_groups = dim ÷ MAX_THREADS
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
    # Added: psi .*= Complex{T2}(-1)
    # psi .*= Complex{T2}(-1)
    result .= psi

    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @metal threads=MAX_THREADS groups=num_groups kernelHx!(psi, mask, result)
    end
    if qaoa.parity_symmetry
        @metal threads=MAX_THREADS groups=num_groups_parity kernelHxParity!(psi, dim, result)
    end
    return nothing
end

function Hx_ψ!(qaoa::QAOA{T1, T2, T3}, psi::AbstractVector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}
    dim = length(psi)
    N = Int(log2(dim))

    num_groups = dim ÷ MAX_THREADS
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
    # Added: psi .*= Complex{T2}(-1)
    # psi .*= Complex{T2}(-1)
    result = copy(psi)

    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @metal threads=MAX_THREADS groups=num_groups kernelHx!(psi, mask, result)
    end
    if qaoa.parity_symmetry
        @metal threads=MAX_THREADS groups=num_groups_parity kernelHxParity!(psi, dim, result)
    end
    return nothing
end

function kernelHCψ!(hc::AbstractVector{Complex{T2}}, psi::AbstractVector{Complex{T2}}) where T2 <: Real
    i = thread_position_in_grid_1d()
    psi[i] *= hc[i]
    return
end

function Hzz_ψ!(qaoa::QAOA{T1, T2, T3}, psi::AbstractVector{Complex{T2}}) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}
    dim = length(psi)
    num_groups = dim ÷ MAX_THREADS

    @metal threads=MAX_THREADS groups=num_groups kernelHCψ!(qaoa.HC, psi)
    return nothing
end

function kernelIm(state::AbstractVector{Complex{T2}}) where T2<:Real
    i=thread_position_in_grid_1d()
    state[i] *= Complex{T2}(-im)
    return
end

function applyQAOALayerDerivative!(qaoa::QAOA{T1, T2, T3}, 
    elem::T2, 
    pos::Int, 
    state::AbstractVector{Complex{T2}}, 
    result::AbstractVector{Complex{T2}}
    ) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}
    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hzz_ψ!(qaoa, state)
    else
        Hx_ψ!(qaoa, state, result)
    end
    
    dim = length(state)
    num_groups = dim ÷ MAX_THREADS
    
    @metal threads=MAX_THREADS groups=num_groups kernelIm(state)
    return nothing
end

function applyQAOALayerDerivative!(qaoa::QAOA{T1, T2, T3}, 
    elem::T2, 
    pos::Int, 
    state::AbstractVector{Complex{T2}}
    ) where {T1<:AbstractGraph, T2<:Real, T3<:METALBackend}

    applyQAOALayer!(qaoa, elem, pos, state)
    if isodd(pos)
        Hzz_ψ!(qaoa, state)
    else
        Hx_ψ!(qaoa, state)
    end
    
    dim = length(state)
    num_groups = dim ÷ MAX_THREADS
    
    @metal threads=MAX_THREADS groups=num_groups kernelIm(state)
    return nothing
end