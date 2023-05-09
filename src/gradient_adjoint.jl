@doc raw"""
    gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real
Algorithm to compute the gradient of the QAOA cost function using adjoint (a reverse-mode) differentiation. 
"""
function gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real
    λ = getQAOAState(qaoa, params)
    ϕ = copy(λ)
    λ .= qaoa.HC .* λ
    μ = similar(λ)
    #costFun = dot(ϕ, λ) |> real
    gradResult = zeros(T, length(params))
    for i ∈ length(params):-1:1
        applyQAOALayerAdjoint!(qaoa, params, i, ϕ)
        μ .= ϕ
        applyQAOALayerDerivative!(qaoa, params, i, μ)
        gradResult[i] = 2.0*real(dot(λ, μ))
        if i > 1
            applyQAOALayerAdjoint!(qaoa, params, i, λ)
        end
    end 
    return gradResult
end

function applyQAOALayerAdjoint!(qaoa::QAOA, params::Vector{T}, pos::Int, state::Vector{Complex{T}}) where T<: Real
    if isodd(pos)
        # γ-type parameter
        state .= exp.(im * params[pos] * qaoa.HC) .* state
    else
        # β-type parameter
        QAOALandscapes.fwht!(state, qaoa.N)
        state .= exp.(im * params[pos] * qaoa.HB) .* state
        QAOALandscapes.ifwht!(state, qaoa.N)
    end
end

function applyQAOALayerDerivative!(qaoa::QAOA, params::Vector{T}, pos::Int, state::Vector{Complex{T}}) where T<: Real
    if isodd(pos)
        # γ-type parameter
        state .= exp.(-im * params[pos] * qaoa.HC) .* state
        state .= (-im .* qaoa.HC) .* state
    else
        # β-type parameter
        QAOALandscapes.fwht!(state, qaoa.N)
        state .= exp.(-im * params[pos] * qaoa.HB) .* state
        state .= (-im .* qaoa.HB) .* state
        QAOALandscapes.ifwht!(state, qaoa.N)
    end
end
