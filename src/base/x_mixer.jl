struct XMixer <: AbstractMixer
    N::Int
end

function (hamX::XMixer)(ψ::Vector{T}) where T <: Complex
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    temp_ψ = copy(ψ)
    for q in 1:N
        mask = 1 << (q - 1)
        for index ∈ 0:(dim-1)
            if (index & mask) == 0
                ψ[index + 1] += temp_ψ[index + 1 + mask]
            else
                ψ[index + 1] += temp_ψ[index + 1 - mask]
            end
        end
    end
    if N+1 == hamX.N
        for l ∈ 1:dim÷2
            ψ[l] += temp_ψ[dim-l+1]
            ψ[dim-l+1] += temp_ψ[l]
        end
    end
    return nothing
end


#### METAL kernels ########
#### kernels for HB|ψ⟩ ####
function kernel_x_mixer!(psi::MtlVector{T}, bitmask::Int, result::MtlVector{T}) where T<:Complex
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

function kernel_x_mixer_parity!(psi::MtlVector{T}, dim::Int, result::MtlVector{T}) where T<:Complex
    i = thread_position_in_grid_1d()
    psi[i]       += result[dim-i+1]
    psi[dim-i+1] += result[i]
    
    return nothing
end

function (hamX::XMixer)(ψ::MtlVector{T}) where T <: Complex
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