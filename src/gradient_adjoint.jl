@doc raw"""
    gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real
Compute the gradient of the QAOA cost function using adjoint (a reverse-mode) differentiation. We implement the algorithm 
proposed in [*this reference*](https://arxiv.org/abs/2009.02823). https://arxiv.org/pdf/2011.02991.pdf
"""
function gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real
    λ = getQAOAState(qaoa, params)
    ϕ = copy(λ)
    if typeof(qaoa.hamiltonian) <: Vector
        λ .= qaoa.hamiltonian .* λ
    else
        λ .= qaoa.hamiltonian * λ
    end
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

function applyQAOALayer!(qaoa::QAOA, params::Vector{T}, pos::Int, state::Vector{Complex{T}}) where T<: Real
    if isodd(pos)
        # γ-type parameter
        state .= exp.(-im * params[pos] * qaoa.HC) .* state
    else
        # β-type parameter
        QAOALandscapes.fwht!(state, qaoa.N)
        state .= exp.(-im * params[pos] * qaoa.HB) .* state
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

@doc raw"""
    geometricTensor(qaoa::QAOA, params::Vector{T}, ψ0::AbstractVector{Complex{T}}) where T<: Real
Compute the geometricTensor of the QAOA cost function using adjoint (a reverse-mode) differentiation. We implement the algorithm 
proposed in [*this reference*](https://arxiv.org/pdf/2011.02991.pdf)
"""
function geometricTensor(qaoa::QAOA, params::Vector{T}, ψ0::AbstractVector{Complex{T}}) where T<: Real
    T_vec = zeros(Complex{T}, length(params))
    L_mat = zeros(Complex{T}, length(params), length(params))
    G_mat = zeros(Complex{T}, length(params), length(params))
    
    χ = copy(ψ0)
    applyQAOALayer!(qaoa, params, 1, χ)

    ψ = copy(χ)
    λ = similar(ψ)
    μ = similar(ψ)

    ϕ = copy(ψ0)
    applyQAOALayerDerivative!(qaoa, params, 1, ϕ)
    
    T_vec[1]    = dot(χ, ϕ)
    L_mat[1, 1] = dot(ϕ, ϕ)
    
    for j ∈ 2:length(params)
        λ .= copy(ψ)
        ϕ .= copy(ψ)
        applyQAOALayerDerivative!(qaoa, params, j, ϕ)

        L_mat[j, j] = dot(ϕ, ϕ)
        for i ∈ j-1:-1:1
            applyQAOALayerAdjoint!(qaoa, params, i+1, ϕ)
            applyQAOALayerAdjoint!(qaoa, params, i, λ)
            μ .= copy(λ)
            applyQAOALayerDerivative!(qaoa, params, i, μ)
            L_mat[i,j] = dot(μ, ϕ)
        end
        T_vec[j] = dot(χ, ϕ)
        applyQAOALayer!(qaoa, params, j, ψ)
    end
    
    for j ∈ eachindex(params)
        for i ∈ eachindex(params)
            if i ≤ j
                G_mat[i,j] = L_mat[i,j] - T_vec[i]' * T_vec[j]
            else
                G_mat[i,j] = L_mat[j,i]' - T_vec[i]' * T_vec[j]
            end
        end
    end

    return G_mat
end