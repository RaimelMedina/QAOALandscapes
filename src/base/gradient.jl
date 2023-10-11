@doc raw"""
    gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real
Compute the gradient of the QAOA cost function using adjoint (a reverse-mode) differentiation. We implement the algorithm 
proposed in [*this reference*](https://arxiv.org/abs/2009.02823). https://arxiv.org/pdf/2011.02991.pdf
"""
function gradCostFunction(qaoa::QAOA, params::AbstractVector{T}) where T<: Real
    # this will update/populate qaoa.state which we will call |λ⟩ following the paper
    λ = getQAOAState(qaoa, params)
    
    # |ϕ⟩ := |λ⟩
    ϕ = copy(λ)

    # |λ⟩ := H |λ⟩
    Hzz_ψ!(qaoa, λ)
    
    # now we allocate |μ⟩
    μ = similar(λ)

    gradResult = zeros(T, length(params))
    
    for i ∈ length(params):-1:1
        # |ϕ⟩ ← (Uᵢ)†|ϕ⟩    
        applyQAOALayer!(qaoa, -params[i], i, ϕ)
        
        # |μ⟩ ← |ϕ⟩
        μ .= ϕ

        # |μ⟩ ← dUᵢ/dθᵢ |μ⟩
        applyQAOALayerDerivative!(qaoa, params[i], i, μ)
        
        # ∇Eᵢ = 2 ℜ ⟨ λ | μ ⟩
        gradResult[i] = 2.0*real(dot(λ, μ))
        if i > 1
            #|λ⟩ ← (Uᵢ)†|λ⟩
            applyQAOALayer!(qaoa, -params[i], i, λ)
        end
    end 
    return gradResult
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
    applyQAOALayer!(qaoa, params[1], 1, χ)

    ψ = copy(χ)
    λ = similar(ψ)
    μ = similar(ψ)

    ϕ = copy(ψ0)
    applyQAOALayerDerivative!(qaoa, params[1], 1, ϕ)
    
    T_vec[1]    = dot(χ, ϕ)
    L_mat[1, 1] = dot(ϕ, ϕ)
    
    for j ∈ 2:length(params)
        λ .= copy(ψ)
        ϕ .= copy(ψ)
        applyQAOALayerDerivative!(qaoa, params[j], j, ϕ)

        L_mat[j, j] = dot(ϕ, ϕ)
        for i ∈ j-1:-1:1
            applyQAOALayer!(qaoa, -params[i+1], i+1, ϕ)
            applyQAOALayer!(qaoa, -params[i], i, λ)
            μ .= copy(λ)
            applyQAOALayerDerivative!(qaoa, params[i], i, μ)
            L_mat[i,j] = dot(μ, ϕ)
        end
        T_vec[j] = dot(χ, ϕ)
        applyQAOALayer!(qaoa, params[j], j, ψ)
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


@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. 
The computation is done analytically since it has proven to be faster than the previous implementation using [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) package
"""
function hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}; diffMode=:manual) where T<:Real
    if diffMode==:forward
        matHessian = ForwardDiff.hessian(qaoa, Γ)
        return matHessian
    elseif diffMode==:manual
        p = length(Γ) ÷ 2
        
        ψ = getQAOAState(qaoa, Γ)
        
        ψCol = similar(ψ)
        ψRow  = similar(ψ)
        ψRowCol  = similar(ψ)
        
        matHessian = zeros(2p, 2p)
        for col in 1:2p
            ψCol .= ∂ψ(qaoa, Γ, col)
            for row in col:2p
                ψRow .= ∂ψ(qaoa, Γ, row)
                ψRowCol .= ∂ψ(qaoa, Γ, col, row)
                
                matHessian[row, col] = 2*real(dot(ψRow, qaoa.HC .* ψCol)) + 2*real(dot(ψ, qaoa.HC .* ψRowCol))
                if col != row
                    matHessian[col, row] = matHessian[row, col]
                end
            end
        end
        return matHessian
    else
        throw(ArgumentError("diffMode=$(diffMode) not supported. Only ':manual' or ':forward' methods are implemented"))
    end
end

function ∂ψ(qaoa::QAOA, Γ::Vector{T}, i::Int) where T<:Real
    ψ = plus_state(T, qaoa.N)
    @inbounds @simd for idx ∈ eachindex(Γ)
        if idx==i
            applyQAOALayerDerivative!(qaoa, Γ[idx], idx, ψ)
        else
            applyQAOALayer!(qaoa, Γ[idx], idx, ψ)
        end
    end
    return ψ
end

function ∂ψ(qaoa::QAOA, Γ::Vector{T}, i::Int, j::Int) where T<:Real
    ψ = plus_state(T, qaoa.N)
    @inbounds @simd for idx ∈ eachindex(Γ)
        if i==j
            if idx==i
                applyQAOALayer!(qaoa, Γ[idx], idx, ψ)
                if isodd(idx)
                    Hzz_ψ!(qaoa, ψ)
                    Hzz_ψ!(qaoa, ψ)
                    ψ .*= -1.0
                else
                    Hx_ψ!(qaoa, ψ)
                    Hx_ψ!(qaoa, ψ)
                    ψ .*= -1.0
                end
            else
                applyQAOALayer!(qaoa, Γ[idx], idx, ψ)
            end
        else
            if idx==i || idx==j
                applyQAOALayerDerivative!(qaoa, Γ[idx], idx, ψ)
            else
                applyQAOALayer!(qaoa, Γ[idx], idx, ψ)
            end
        end
    end
    return ψ
end

function hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}, idx::Vector{Int64}) where T<:Real
    ψ = getQAOAState(qaoa, Γ)
    ψRow    = ∂ψ(qaoa, Γ, idx[1])
    ψCol    = ∂ψ(qaoa, Γ, idx[2])
    ψRowCol = ∂ψ(qaoa, Γ, idx[1], idx[2])

    hessianElement = 2*real(dot(ψRow, qaoa.HC .* ψCol)) + 2*real(dot(ψ, qaoa.HC .* ψRowCol))
    return hessianElement
end


"""
    getHessianIndex(qaoa::QAOA, Γ::AbstractVector{T}; checks=true, tol=1e-6) where T<:Real

Calculate the Hessian index of a stationary (it checks the gradient norm) point of the QAOA energy function

# Arguments
- `qaoa`: a QAOA object.
- `Γ`: a vector of parameters.

# Keyword Arguments
- `checks=true`: a boolean to decide whether to check if the gradient of the cost function is smaller than a certain tolerance.
- `tol=1e-6`: a tolerance level for the gradient of the cost function.

# Output
- Returns the Hessian index, i.e., the number of negative eigenvalues of the Hessian matrix.

# Notes
The function first calculates the gradient of the cost function for the given `qaoa` and `Γ`. If `checks=true`, it asserts that the norm of this gradient is less than `tol`. It then calculates the Hessian matrix and its eigenvalues, and returns the count of eigenvalues less than zero.

"""
function getHessianIndex(qaoa::QAOA, Γ::AbstractVector{T}; checks=true, tol=1e-6) where T<:Real
    checks && norm(gradCostFunction(qaoa, Γ)) < tol : nothing : @warn "Gradient norm is above the tolerance threshold. Check convergence"
    
    hessian_eigvals = hessianCostFunction(qaoa, Γ) |> eigvals
    return count(x->x<0, filter(x -> abs(x) > tol, hessian_eigvals))
end