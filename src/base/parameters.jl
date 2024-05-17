mutable struct Parameter{T<:Real} <: AbstractVector{T}
    value::T
    data::Vector{T}
end

Base.getindex(p::Parameter, i::Int) = p.data[i]
Base.size(p::Parameter) = size(p.data)
Base.length(p::Parameter) = length(p.data)
Base.setindex!(p::Parameter{T}, v::T, i::Int) where T<:Real = (p.data[i]=v)

function setvalue!(param::Parameter{T}, qaoa::QAOA{P, H, M}) where {P, H, M, T<:Real}
    param.value = qaoa(param)
end
function setvalue!(param::Parameter{T}, val::T) where T<:Real
    param.value = val
end

Parameter(vec::Vector{T}) where T<:Real = Parameter(T(0), vec)

(qaoa::QAOA{P, H, M})(param::Parameter{T}) where {P, H, M, T<:Real} = qaoa(param.data)


@doc raw"""
    toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})

Implements the symmetries of the QAOA for different graphs. For more detail see the following [`reference`](https://arxiv.org/abs/2209.01159).

For an arbitrary graph, we can restrict both ``\gamma`` and ``\beta`` parameters to the ``[-\pi/2, \pi/2]`` interval. Furthermore, ``\beta`` parameters
can be restricted even further to the ``[-\pi/4, \pi/4]`` interval (see [`here`](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067))
Finally, when dealing with regular graphs with odd degree `\gamma` paramaters can be brought to the ``[-\pi/4, \pi/4]`` interval.
This function modifies inplace the initial input vector ``Γ``. 
"""
function toFundamentalRegion!(qaoa::QAOA{P, H, M}, 
    Γ::AbstractVector{T}
    ) where {P, H, M, T<:Real}
    
    p = length(Γ) ÷ 2
    β = view(Γ, 2:2:2p)
    γ = view(Γ, 1:2:2p)

    problem_degree = qaoa.problem.degree
    isWeightedG    = qaoa.problem.weightedQ
    isZ2invariant  = qaoa.problem.z2_sym

    # When HB-> XMixer then we know that βₗ ∈ [-π/2, π/2]
    # If HC is not weighted then we can also restrict γₗ ∈ [-π/2, π/2]
    β .= mod.(β, π) .|> T
    β[β .> π/2] .-= T(π)
    if !isWeightedG
        γ .= mod.(γ, π) .|> T
        γ[γ .> π/2] .-= T(π)
    end
    if isZ2invariant
        β .= mod.(β, π/2) .|> T
        β[β .> π/4] .-= T(π/2)
    end
    if !isnothing(problem_degree) && reduce(*, isodd.(problem_degree)) && !isWeightedG
        for i=1:p
            if abs(γ[i]) > π/4 # now folding them even more: to -pi/4, pi/4 interval
                β[i:end] .*= -1 # this requires sign flip of betas!
                γ[i] -= sign(γ[i])*π/2 |> T
            end
        end
    end
    if γ[1] < 0 # making angle gamma_1 positive
        β .*= -1 # by changing the sign of ALL angles
        γ .*= -1
    end
    return nothing
end