mutable struct Parameter{T<:Real} <: AbstractVector{T}
    value::T
    data::Vector{T}
end

Base.getindex(p::Parameter, i::Int) = p.data[i]
Base.size(p::Parameter) = size(p.data)
Base.length(p::Parameter) = length(p.data)
Base.setindex!(p::Parameter{T}, v::T, i::Int) where T<:Real = (p.data[i] = v)

function setvalue!(param::Parameter{T}, qaoa::QAOA) where {T<:Real}
    param.value = qaoa(param.data)
end
function setvalue!(param::Parameter{T}, val::T) where T<:Real
    param.value = val
end

Parameter(vec::Vector{T}) where T<:Real = Parameter(T(0), vec)

(qaoa::QAOA)(param::Parameter{T}) where {T<:Real} = qaoa(param.data)


@doc raw"""
    toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})

Implements the symmetries of the QAOA for different graphs. For more detail see the following [`reference`](https://arxiv.org/abs/2209.01159).

For an arbitrary graph, we can restrict both ``\gamma`` and ``\beta`` parameters to the ``[-\pi/2, \pi/2]`` interval. Furthermore, ``\beta`` parameters
can be restricted even further to the ``[-\pi/4, \pi/4]`` interval (see [`here`](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067))
Finally, when dealing with regular graphs with odd degree `\gamma` paramaters can be brought to the ``[-\pi/4, \pi/4]`` interval.
This function modifies inplace the initial input vector ``Γ``. 
"""
function toFundamentalRegion!(qaoa::QAOA, 
    Γ::AbstractVector{T}
    ) where {T<:Real}
    
    p = length(Γ) ÷ 2
    β = view(Γ, 2:2:2p)
    γ = view(Γ, 1:2:2p)

    problem_degree = qaoa.problem.degree
    isWeightedG    = qaoa.problem.weightedQ
    locality_of_terms  = qaoa.problem.locality
    
    if foldl(&, iseven.(locality_of_terms))
        mixer_and_cost = :commute
    elseif foldl(&, isodd.(locality_of_terms))
        mixer_and_cost = :anticommute
    else
        mixer_and_cost = :nothing
    end

    # When HB-> XMixer then we know that βₗ ∈ [-π/2, π/2)
    if typeof(qaoa.mixer) <: XMixer
        β .= mod.(β, π) .|> T
        β[β .>= π/2] .-= T(π)
    end
    # When HC is not weighted then we can restrict γₗ ∈ [-π/2, π/2)
    if !isWeightedG
        # println("Reducing γ parameters to: [-π/2, π/2)")
        γ .= mod.(γ, π) .|> T
        γ[γ .>= π/2] .-= T(π)
    end

    # Until here all is beautiful and easy to understand

    # When HC is Z₂ symmetric then performing exp(-i (β + π/2) HB) ∼ exp(-i β HB) (i σˣ)ⁿ
    # does not affects the energy. With this, we fold β's to [-π/4, π/4)
    if typeof(qaoa.mixer) <: XMixer
        if mixer_and_cost == :commute
            β .= mod.(β, π/2) .|> T
            β[β .>= π/4] .-= T(π/2)
        elseif mixer_and_cost == :anticommute
            for i ∈ 1:p
                if β[i] < -π/4 || β[i] ≥ π/4
                    γ[1:i] .*= -1
                    β[i] -= sign(β[i])*π/2 |> T
                    for j in 1:p
                        if γ[j] == π/4
                           γ[j] *= -1
                        end
                    end
                end
            end
        end
    end

    if !isnothing(problem_degree) && reduce(*, isodd.(problem_degree)) && !isWeightedG
        # println("Reducing γ parameters to: [-π/4, π/4) affecting β indices")
        for i=1:p
            if γ[i] < -π/4 || γ[i] ≥ π/4 
                β[i:end] .*= -1 # this requires sign flip of betas!
                γ[i] -= sign(γ[i])*π/2 |> T
                for j in 1:p
                    if β[j] == π/4
                       β[j] *= -1
                    end
                end 
            end
        end
    end

    if !isnothing(problem_degree) && reduce(*, iseven.(problem_degree)) && !isWeightedG
        # println("Reducing γ parameters to: [-π/4, π/4)")
        γ .= mod.(γ, π/2) .|> T
        γ[γ .>= π/4] .-= T(π/2)
    end
    if γ[1] < 0 # making angle gamma_1 positive
        # println("γ₁ negative -> flipping sign of all paramaters")
        β .*= -1 # by changing the sign of ALL angles
        γ .*= -1
    end
    return nothing
end