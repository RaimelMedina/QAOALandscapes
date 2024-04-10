function kernelExpX!(psi::AbstractGPUArray{T}, bitmask::Int, cos_a::K, sin_a::K) where {T, K}
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

function kernelExpXParity!(psi::AbstractGPUArray{T}, dim::Int, cβ::K, sβ::K) where {T, K}
    i = thread_position_in_grid_1d()
    val1 = psi[i]
    val2 = psi[dim-i+1]

    psi[i]       = cβ * val1 - im * sβ * val2
    psi[dim-i+1] = cβ * val2 - im * sβ * val1
    
    return
end

function applyExpX!(psi::AbstractGPUArray{T}, k::Int, cos_a::K, sin_a::K) where {T,K}
    dim = length(psi)
    bitmask = 1 << (k-1)
    num_groups = dim ÷ MAX_THREADS
    @metal threads=MAX_THREADS groups=num_groups kernelExpX!(psi, bitmask, cos_a, sin_a)
    return nothing
end

function kernelExpHC!(hc::AbstractGPUArray{T}, ψ::AbstractGPUArray{K}, γ::R) where {T, K, R}
    i = thread_position_in_grid_1d()
    ψ[i] *= exp(-im * γ * hc[i])
    return
end

function applyExpLayer!(hc::AbstractGPUArray{T}, ψ::AbstractGPUArray{K}, γ::R) where {T, K, R}
    dim = length(ψ)
    num_groups = dim ÷ MAX_THREADS
    @metal threads=MAX_THREADS groups=num_groups kernelExpHC!(hc, ψ, γ)
    return nothing
end


function kernelHCψ!(hc::AbstractVector{T}, psi::AbstractVector{R}) where {T, R}
    i = thread_position_in_grid_1d()
    psi[i] *= hc[i]
    return
end


function Hc_ψ!(ham::AbstractVector{S}, ψ::AbstractVector{T}) where {S, T}
    dim = length(ψ)
    num_groups = dim ÷ MAX_THREADS

    @metal threads=MAX_THREADS groups=num_groups kernelHCψ!(ham, ψ)
    return nothing
end

#### METAL kernels ########
#### kernels for HB|ψ⟩ ####
function kernel_x_mixer!(psi::AbstractGPUArray{T}, bitmask::Int, result::AbstractGPUArray{T}) where T<:Complex
    index = thread_position_in_grid_1d() - 1
    i1 = index + 1
    if index & bitmask == 0
        i2 = index + 1 + bitmask
    else
        i2 = index + 1 - bitmask
    end
    psi[i1] += result[i2]
    return nothing
end

function kernel_x_mixer_parity!(psi::AbstractGPUArray{T}, dim::Int, result::AbstractGPUArray{T}) where T<:Complex
    i = thread_position_in_grid_1d()
    psi[i]       += result[dim-i+1]
    psi[dim-i+1] += result[i]
    
    return nothing
end

function (hamX::XMixer)(ψ::AbstractGPUArray{T}, temp_ψ::AbstractGPUArray{T}) where T <: Complex
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    num_groups = dim ÷ MAX_THREADS
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
    #temp_ψ::MtlVector{T} = copy(ψ)

    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @metal threads=MAX_THREADS groups=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
    end
    if N+1 == hamX.N
        @metal threads=MAX_THREADS groups=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
    end
    return nothing
end

function (hamX::XMixer)(ψ::AbstractGPUArray{T}) where T <: Complex
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    num_groups = dim ÷ MAX_THREADS
    num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
    temp_ψ::MtlVector{T} = copy(ψ)

    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @metal threads=MAX_THREADS groups=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
    end
    if N+1 == hamX.N
        @metal threads=MAX_THREADS groups=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
    end
    return nothing
end

function applyExpLayer!(mixer::XMixer, psi::AbstractGPUArray{T}, β::R) where {T, R}
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