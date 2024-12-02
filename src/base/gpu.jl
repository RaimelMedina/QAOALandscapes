function kernelExpX!(psi, bitmask::Int, cos_a::K, sin_a::K) where {K}
    index = threadIdx().x - 1
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

function kernelExpXParity!(psi, dim::Int, cβ::K, sβ::K) where {K}
    i = threadIdx().x
    val1 = psi[i]
    val2 = psi[dim-i+1]

    psi[i]       = cβ * val1 - im * sβ * val2
    psi[dim-i+1] = cβ * val2 - im * sβ * val1
    
    return
end

function applyExpX!(psi::T, k::Int, cos_a::K, sin_a::K) where {T<:AbstractGPUVector,K}
    dim = length(psi)
    bitmask = 1 << (k-1)

    threads = min(MAX_THREADS, dim)
    num_groups = cld(dim, threads)
    
    @cuda threads=threads blocks=num_groups kernelExpX!(psi, bitmask, cos_a, sin_a)
    return nothing
end

function kernelExpHC!(hc, ψ, γ::R) where {R}
    i = threadIdx().x
    ψ[i] *= exp(-im * γ * hc[i])
    return
end

function applyExpLayer!(hc::T, ψ::K, γ::R) where {T<:AbstractGPUVector, K<:AbstractGPUVector, R}
    dim = length(ψ)
    
    threads = min(MAX_THREADS, dim)
    num_groups = cld(dim, threads)

    @cuda threads=threads blocks=num_groups kernelExpHC!(hc, ψ, γ)
    return nothing
end


function kernelHCψ!(hc, psi)
    i = threadIdx().x
    psi[i] *= hc[i]
    return nothing
end


function Hc_ψ!(ham::S, ψ::T) where {S<:AbstractGPUVector, T<:AbstractGPUVector}
    dim = length(ψ)

    threads = min(MAX_THREADS, dim)
    num_groups = cld(dim, threads)

    @cuda threads=threads blocks=num_groups kernelHCψ!(ham, ψ)
    return nothing
end

#### kernels for HB|ψ⟩ ####
function kernel_x_mixer!(psi::T, bitmask::Int, result::T) where T<:AbstractGPUVector
    index = threadIdx().x - 1
    i1 = index + 1
    if index & bitmask == 0
        i2 = index + 1 + bitmask
    else
        i2 = index + 1 - bitmask
    end
    psi[i1] += result[i2]
    return nothing
end

function kernel_x_mixer_parity!(psi, dim::Int, result)
    i = threadIdx().x
    psi[i]       += result[dim-i+1]
    psi[dim-i+1] += result[i]
    
    return nothing
end

function (hamX::XMixer)(ψ::T, temp_ψ::T) where T <: AbstractGPUVector
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    threads = min(MAX_THREADS, dim)
    num_groups = cld(dim, threads)
    num_groups_parity = cld(dim ÷ 2, threads)
    
    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @cuda threads=threads blocks=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
    end
    if N+1 == hamX.N
        @cuda threads=threads blocks=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
    end
    return nothing
end

function (hamX::XMixer)(ψ::T) where T <: AbstractGPUVector
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    threads = min(MAX_THREADS, dim)
    num_groups = cld(dim, threads)
    num_groups_parity = cld(dim ÷ 2, threads)
    
    temp_ψ = copy(ψ)

    for qubit in 1:N
        mask = 1 << (qubit - 1)
        @cuda threads=threads blocks=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
    end
    if N+1 == hamX.N
        @cuda threads=threads blocks=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
    end
    return nothing
end

function applyExpLayer!(mixer::XMixer, psi::T, β::R) where {T<:AbstractGPUVector, R}
    cβ = cos(β)
    sβ = sin(β)
    
    dim = length(psi)
    N = Int(log2(dim))
    
    threads = min(MAX_THREADS, dim)
    num_groups_parity = cld(dim ÷ 2, threads)

    # Loop over spins
    for i ∈ 1:N
        applyExpX!(psi, i, cβ, sβ)
    end
    # check if there is parity symmetry
    if N+1 == mixer.N # Z2 symmetric case
        @cuda threads=threads blocks=num_groups_parity kernelExpXParity!(psi, dim, cβ, sβ)
    end
    return nothing
end


function hamiltonian_cuda(cp::ClassicalProblem{T}, sym_sector = true) where T
    # Precompute constants and parameters
    interactions = cp.interactions
    
    # Extract keys (terms) and values (coefficients) from interactions
    interaction_keys = collect(keys(interactions))
    interaction_values = collect(values(interactions))
    n_interactions = length(interaction_keys)
    
    # Flatten the keys into arrays
    flattened_terms = Int[]
    for term in interaction_keys
        append!(flattened_terms, term)
    end
    
    term_starts = Int[1]
    running_sum = 1
    for term in interaction_keys
        running_sum += length(term)
        push!(term_starts, running_sum)
    end
    
    # Check Z2 symmetry
    z2_sym = all(iseven, cp.locality)
    dim = if sym_sector && z2_sym
        2^(cp.n-1)
    else
        sym_sector && !z2_sym && @info "Problem is not symmetric"
        2^cp.n
    end
    
    # Data goes into the GPU - careful with types
    d_flattened_terms = CuArray{Int32}(flattened_terms)
    d_term_starts = CuArray{Int32}(term_starts)
    d_values = CuArray{T}(interaction_values)
    d_ham = CUDA.zeros(Complex{T}, dim)
    
    function hamiltonian_kernel!(ham, flattened_terms, term_starts, values, n_interactions, dim)
        idx = (threadIdx().x + (blockIdx().x - 1) * blockDim().x)
        
        if idx <= dim
            energy = zero(Complex{T})
            
            @inbounds for i in 1:n_interactions
                start_idx = term_starts[i]
                end_idx = term_starts[i + 1] - 1
                
                parity = Int32(0)
                @inbounds for j in start_idx:end_idx
                    qubit = flattened_terms[j]
                    parity = parity ⊻ (((idx - 1) >> (qubit - 1)) & 1)
                end
                
                energy += values[i] * (1 - 2*parity)
            end
            
            @inbounds ham[idx] = energy
        end
        
        return nothing
    end
    
    # Launch kernel
    threads = min(MAX_THREADS, dim)
    blocks = cld(dim, threads)
    
    @cuda threads=threads blocks=blocks hamiltonian_kernel!(
        d_ham, d_flattened_terms, d_term_starts, d_values, n_interactions, dim
    )
    
    CUDA.synchronize()
    return d_ham
end

# ##### CUDA KERNELS ######

# function kernelExpX!(psi::AbstractGPUArray{T}, bitmask::Int, cos_a::K, sin_a::K) where {T, K}
#     # index = thread_position_in_grid_1d() - 1
#     index = threadIdx().x - 1
#     if index & bitmask == 0
#         i1 = index + 1
#         i2 = index + 1 + bitmask

#         # Create local copies to avoid race conditions
#         val1 = psi[i1]
#         val2 = psi[i2]

#         psi[i1] = cos_a * val1 - im * sin_a * val2
#         psi[i2] = cos_a * val2 - im * sin_a * val1
#     end
#     return
# end

# function kernelExpXParity!(psi::AbstractGPUArray{T}, dim::Int, cβ::K, sβ::K) where {T, K}
#     # i = thread_position_in_grid_1d()
#     i = threadIdx().x
#     val1 = psi[i]
#     val2 = psi[dim-i+1]

#     psi[i]       = cβ * val1 - im * sβ * val2
#     psi[dim-i+1] = cβ * val2 - im * sβ * val1
    
#     return
# end

# function applyExpX!(psi::AbstractGPUArray{T}, k::Int, cos_a::K, sin_a::K) where {T,K}
#     dim = length(psi)
#     bitmask = 1 << (k-1)
#     num_groups = dim ÷ MAX_THREADS
#     @cuda threads=MAX_THREADS blocks=num_groups kernelExpX!(psi, bitmask, cos_a, sin_a)
#     return nothing
# end

# function kernelExpHC!(hc::AbstractGPUArray{T}, ψ::AbstractGPUArray{K}, γ::R) where {T, K, R}
#     # i = thread_position_in_grid_1d()
#     i = threadIdx().x
#     ψ[i] *= exp(-im * γ * hc[i])
#     return
# end

# function applyExpLayer!(hc::AbstractGPUArray{T}, ψ::AbstractGPUArray{K}, γ::R) where {T, K, R}
#     dim = length(ψ)
#     num_groups = dim ÷ MAX_THREADS
#     @cuda threads=MAX_THREADS blocks=num_groups kernelExpHC!(hc, ψ, γ)
#     return nothing
# end


# function kernelHCψ!(hc::AbstractGPUArray{T}, psi::AbstractGPUArray{R}) where {T, R}
#     # i = thread_position_in_grid_1d()
#     i = threadIdx().x
#     psi[i] *= hc[i]
#     return
# end


# function Hc_ψ!(ham::AbstractGPUArray{S}, ψ::AbstractGPUArray{T}) where {S, T}
#     dim = length(ψ)
#     num_groups = dim ÷ MAX_THREADS

#     @cuda threads=MAX_THREADS blocks=num_groups kernelHCψ!(ham, ψ)
#     return nothing
# end

# #### METAL kernels ########
# #### kernels for HB|ψ⟩ ####
# function kernel_x_mixer!(psi::AbstractGPUArray{T}, bitmask::Int, result::AbstractGPUArray{T}) where T<:Complex
#     # index = thread_position_in_grid_1d() - 1
#     index = threadIdx().x - 1
#     i1 = index + 1
#     if index & bitmask == 0
#         i2 = index + 1 + bitmask
#     else
#         i2 = index + 1 - bitmask
#     end
#     psi[i1] += result[i2]
#     return nothing
# end

# function kernel_x_mixer_parity!(psi::AbstractGPUArray{T}, dim::Int, result::AbstractGPUArray{T}) where T<:Complex
#     # i = thread_position_in_grid_1d()
#     i = threadIdx().x
#     psi[i]       += result[dim-i+1]
#     psi[dim-i+1] += result[i]
    
#     return nothing
# end

# function (hamX::XMixer)(ψ::AbstractGPUArray{T}, temp_ψ::AbstractGPUArray{T}) where T <: Complex
#     dim = length(ψ)
#     N = dim |> log2 |> Int
#     @assert N == hamX.N || N + 1 == hamX.N

#     num_groups = dim ÷ MAX_THREADS
#     num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
#     #temp_ψ::MtlVector{T} = copy(ψ)

#     for qubit in 1:N
#         mask = 1 << (qubit - 1)
#         @cuda threads=MAX_THREADS blocks=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
#     end
#     if N+1 == hamX.N
#         @cuda threads=MAX_THREADS blocks=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
#     end
#     return nothing
# end

# function (hamX::XMixer)(ψ::AbstractGPUArray{T}) where T <: Complex
#     dim = length(ψ)
#     N = dim |> log2 |> Int
#     @assert N == hamX.N || N + 1 == hamX.N

#     num_groups = dim ÷ MAX_THREADS
#     num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS
    
#     temp_ψ::AbstractGPUArray{T} = copy(ψ)

#     for qubit in 1:N
#         mask = 1 << (qubit - 1)
#         @cuda threads=MAX_THREADS blocks=num_groups kernel_x_mixer!(ψ, mask, temp_ψ)
#     end
#     if N+1 == hamX.N
#         @cuda threads=MAX_THREADS blocks=num_groups_parity kernel_x_mixer_parity!(ψ, dim, temp_ψ)
#     end
#     return nothing
# end

# function applyExpLayer!(mixer::XMixer, psi::AbstractGPUArray{T}, β::R) where {T, R}
#     cβ = cos(β)
#     sβ = sin(β)
    
#     dim = length(psi)
#     N = Int(log2(dim))
    
#     num_groups_parity = (dim ÷ 2) ÷ MAX_THREADS

#     # Loop over spins
#     for i ∈ 1:N
#         applyExpX!(psi, i, cβ, sβ)
#     end
#     # check if there is parity symmetry
#     if N+1 == mixer.N # Z2 symmetric case
#         @cuda threads=MAX_THREADS blocks=num_groups_parity kernelExpXParity!(psi, dim, cβ, sβ)
#     end
#     return nothing
# end