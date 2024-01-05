mutable struct Parameter{T<:Real} <: AbstractVector{T}
    value::T
    data::Vector{T}
end

Base.getindex(p::Parameter, i::Int) = p.data[i]
Base.size(p::Parameter) = size(p.data)
Base.length(p::Parameter) = length(p.data)
Base.setindex!(p::Parameter{T}, v::T, i::Int) where T<:Real = (p.data[i]=v)

function setvalue!(param::Parameter{T}, qaoa::QAOA{G, T, B}) where {G<:AbstractGraph, T<:Real, B<:AbstractBackend}
    param.value = qaoa(param)
end
function setvalue!(param::Parameter{T}, val::T) where T<:Real
    param.value = val
end

Parameter(vec::Vector{T}) where T<:Real = Parameter(T(0), vec)

(qaoa::QAOA{G, T, B})(param::Parameter{T}) where {G<:AbstractGraph, T<:Real, B<:AbstractBackend} = qaoa(param.data)


@doc raw"""
    toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})

Implements the symmetries of the QAOA for different graphs. For more detail see the following [`reference`](https://arxiv.org/abs/2209.01159).

For an arbitrary graph, we can restrict both ``\gamma`` and ``\beta`` parameters to the ``[-\pi/2, \pi/2]`` interval. Furthermore, ``\beta`` parameters
can be restricted even further to the ``[-\pi/4, \pi/4]`` interval (see [`here`](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067))
Finally, when dealing with regular graphs with odd degree `\gamma` paramaters can be brought to the ``[-\pi/4, \pi/4]`` interval.
This function modifies inplace the initial input vector ``Γ``. 
"""
function toFundamentalRegion!(qaoa::QAOA{T1, T, T3}, 
    Γ::Vector{T}
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    p = length(Γ) ÷ 2
    β = view(Γ, 2:2:2p)
    γ = view(Γ, 1:2:2p)
    
    # First, folding β ∈ [-π/4, π/4]
    for i=1:p #beta angles come first, they are between -pi/4, pi/4
        β[i] = T(mod(β[i], π/2)) # folding beta to interval 0, pi/2
        if β[i] > π/4 # translating it to -pi/4, pi/4 interval
            β[i] -= T(π/2)
        end
    end

    graph_degree = degree(qaoa.graph)
    isWeightedG  = typeof(qaoa.graph) <: SimpleWeightedGraph
    if reduce(*, isodd.(graph_degree)) && !isWeightedG 
        # enter here if each vertex has odd degree d. Assuming regular graphs here :|
        # also, this only works for unweighted d-regular random graphs 
        for i=1:p
            γ[i] = T(mod(γ[i], π))  # processing gammas by folding them to -pi/2, pi/2 interval
            if γ[i] > π/2
                γ[i] -=T(π)
            end
            if abs(γ[i]) > π/4 # now folding them even more: to -pi/4, pi/4 interval
                β[i:end] .*= T(-1) # this requires sign flip of betas!
                γ[i] -= T(sign(γ[i])*π/2)
            end
            if γ[1] < 0 # making angle gamma_1 positive
                β .*= T(-1)  # by changing the sign of ALL angles
                γ .*= T(-1)
            end
        end
    else
        for i=1:p
            γ[i] = T(mod(γ[i] + π, 2π) - π)
        end
    end
    return nothing
end