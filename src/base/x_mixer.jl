struct XMixer <: AbstractMixer
    N::Int
end

function (hamX::XMixer)(ψ::Vector{T}) where T <: Complex
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N
    
    temp_ψ = copy(ψ)
    ψ .= T(0)

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
            ψ[l]       += temp_ψ[dim-l+1]
            ψ[dim-l+1] += temp_ψ[l]
        end
    end
    return nothing
end

function (hamX::XMixer)(ψ::Vector{T}, temp_ψ::Vector{T}) where T <: Complex
    dim = length(ψ)
    N = dim |> log2 |> Int
    @assert N == hamX.N || N + 1 == hamX.N

    temp_ψ .= ψ
    ψ .= T(0)
    
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
            ψ[l]       += temp_ψ[dim-l+1]
            ψ[dim-l+1] += temp_ψ[l]
        end
    end
    return nothing
end