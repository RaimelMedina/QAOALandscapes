function elementHessianCostFunction(qaoa::QAOA, Γ::Vector{Float64}, idx::Vector{Int64}; ϵ=cbrt(eps(Float64)))
    γIdx, βIdx = idx[1], idx[2]
    p   = length(Γ) ÷ 2      
    uγ  = _onehot(γIdx, 2p)*ϵ
    uβ  = _onehot(βIdx, 2p)*ϵ 

    if γIdx==βIdx
        hessianElement = (
            -1.0*qaoa(Γ+2uβ)-
            1.0*qaoa(Γ-2uβ)+
            16*qaoa(Γ+uβ)+
            16*qaoa(Γ-uβ) -
            30*qaoa(Γ)
        )/(12.0*ϵ^2)
    else
        hessianElement = (
            -1.0*qaoa(Γ+2uβ+2uγ) + 16*qaoa(Γ+uβ+uγ)+
            1.0*qaoa(Γ+2uβ-2uγ) - 16*qaoa(Γ+uβ-uγ)+
            1.0*qaoa(Γ-2uβ+2uγ) - 16*qaoa(Γ-uβ+uγ)-
            1.0*qaoa(Γ-2uβ-2uγ) + 16*qaoa(Γ-uβ-uγ)
        )/(48.0*ϵ^2)
    end
    return hessianElement
end

@doc raw"""
    getNegativeHessianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric", ϵ=cbrt(eps(Float64)))

Computes the approximation to the minimum (negative) eigenvalue of the Hessian at the TS obtained by padding with zeros
the local minimum `Γmin`. The transition state is completely specified by the index of the γ component `ig`, and the 
type of transition states (`"symmetric"` or `"non_symmetric"`). The cost of obtaining this approximate eigenvalue is basically
the cost of computing two matrix elements of a Hessian.
"""
function getNegativeHessianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric", ϵ=cbrt(eps(Float64)))
    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    p    = length(Γmin) ÷ 2
    
    γIdx = 1:2:(2p+2)
    βIdx = 2:2:(2p+2)

    b    = 0.0
    bbar = 0.0

    if tsType=="symmetric"
        b = elementHessianCostFunction(qaoa, ΓTs, [γIdx[ig], βIdx[ig]], ϵ=ϵ)
        if (ig != 1 && ig != p+1)
            bbar = b - elementHessianCostFunction(qaoa, Γmin, [γIdx[ig], βIdx[ig-1]])
            bbar /= 2
        else      
            bbar = b/sqrt(2)
        end
    elseif tsType=="non_symmetric"
        b = elementHessianCostFunction(qaoa, ΓTs, [γIdx[ig], βIdx[ig-1]], ϵ=ϵ)
        bbar = b - elementHessianCostFunction(qaoa, Γmin, [γIdx[ig-1], βIdx[ig-1]])
        bbar /= 2
    end
    return b, bbar
end

@doc raw"""
    permuteHessian(depth::Int, i::Int; tsType="symmetric")

Computes the permutation that takes the Hessian at a particular transition state into the form described in the paper
Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.

# Arguments
* `depth::Int`: Circuit depth of the transition state.
* `i::Int`: Index of the γ component at which the zero is added. If `tsType='symmetric'` then `β=i`, otherwise if `tsType='non_symmetric'` `β=i-1`.

# Return
* `listOfIndices::Vector{Int64}`: List of indices correponding to the arrangement of the new basis elements.
* `permMat::Matrix{Float64}`: Matrix implementing the desired permutation.
"""
function permuteHessian(depth::Int, i::Int; tsType="symmetric")
    dim = 2*depth

    γIdx = 1:2:dim
    βIdx = 2:2:dim

    lastIndices = zeros(2)

    if tsType == "symmetric"
        lastIndices = [γIdx[i], βIdx[i]] 
    elseif tsType == "non_symmetric"
        lastIndices = [βIdx[i-1], γIdx[i]] 
    else
        throw(ArgumentError("Only 'symmetric' and 'non_symmetric' values are accepted"))
    end

    listOfIndices = filter(x->x != lastIndices[1] && x != lastIndices[2], 1:dim)
    append!(listOfIndices, lastIndices)

    return listOfIndices, Matrix{Float64}(I(dim))[listOfIndices, :]
end

@doc raw"""
    permuteHessian(H::AbstractArray{Float64,2}, i::Int; tsType="symmetric")

Computes the permutation that takes the Hessian at a particular transition state into the form described in the paper
Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.

# Arguments
* `H::AbstractArray{Float64,2}`: Hessian at the transition state in the original basis.
* `i::Int`: Index of the γ component at which the zero is added. If `tsType='symmetric'` then `β=i`, otherwise if `tsType='non_symmetric'` `β=i-1`.

# Return
* `listOfIndices::Vector{Int64}`: List of indices correponding to the arrangement of the new basis elements.
* `permMat::Matrix{Float64}`: Matrix implementing the desired permutation.
* `HTransformed::Matrix{Float64}`: Transformed Hessian at the transition state. Specifically, we have that ``H_{\mathop{\rm{perm}}}=PHP^{-1}``.
"""
function permuteHessian(H::AbstractArray{Float64,2}, i::Int; tsType="symmetric", returnHessian=false)
    dim = size(H)[1]

    listOfIndices, permutationMat = permuteHessian(dim ÷ 2, i; tsType=tsType)
    return listOfIndices, permutationMat, H[listOfIndices, listOfIndices]
end

@doc raw"""
    getNegativeHessianEigvec(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric", doChecks=false)

Computes the approximate eigenvalue and index-1 eigenvector of the Hessian at the transition state obtained from the local minimum 
`Γmin`. It is completely specified by the parameters `iγ` and `tsType="symmetric"`. If the optional parameter `doChecks=false`
is set to `true`, then the function also returns the relative error in estimating the true eigenvalue as well as the inner product between
the approximate and true eigenvector

# Arguments 
* `qaoa::QAOA`: QAOA object
* `Γmin::Vector{Float64}`: The vector corresponding to a local minimum of QAOAₚ. 
* `ig::Int`: Index of the γ component where the zeros are inserted. 
* `tsType="symmetric"`: In this case, the index of the β component is equal to `ig`. Otherwise, the β index is `ig-1`.

# Optional arguments
* `doChecks=false`: In this case the function returns a dictionary with keys `eigvec_approx` and `eigval_approx`. If set to true it has additional keys => `change_basis`, `eigvec_fidelity` and `eigval_error`

# Returns
* `result::Dict` Dictionary with the following keys: `eigvec_approx`, `eigval_approx`. If `doChecks=true` the following additional keys are available: `change_basis`, `eigvec_fidelity` and `eigval_error`
"""
function getNegativeHessianEigvec(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric", doChecks=false)
    p   = length(Γmin) ÷ 2
    dim = 2(p+1)

    γIdx = 1:2:dim
    βIdx = 2:2:dim

    RowTransform = Matrix{Int64}(I(dim))
    ColTransform = Matrix{Float64}(I(dim))
    vApproximate = zeros(dim)

    #now compute the approximation to the eigenvalue :) 
    b, bbar = getNegativeHessianEigval(qaoa, Γmin, ig; tsType=tsType)
    #we use bbar, or more specifically its sign to determine
    #the approximate eigenvector

    if tsType == "symmetric"
        if (ig != p+1) && (ig != 1)
            RowTransform[dim-1, γIdx[ig]]   = -1
            RowTransform[dim  , βIdx[ig-1]] = -1
        
            ColTransform[γIdx[ig], dim-1]   = -1/2
            ColTransform[βIdx[ig-1], dim  ] = -1/2
            
            vApproximate[dim-1], vApproximate[dim] = 1/sqrt(2), -sign(bbar)/sqrt(2)
        elseif ig==p+1

            RowTransform[dim  , βIdx[ig-1]] = -1
            ColTransform[βIdx[ig-1], dim] = -1/2
            
            vApproximate[dim-1], vApproximate[dim] = -sign(bbar)/sqrt(3), sqrt(2/3)
        else
            RowTransform[dim-1  , γIdx[ig]] = -1
            ColTransform[γIdx[ig], dim-1] = -1/2
            
            vApproximate[dim-1], vApproximate[dim] = -sign(bbar)*sqrt(2/3), 1/sqrt(3)
         end
        
    elseif tsType == "non_symmetric"
        if ig==1
            throw(ArgumentError("Index of the gamma parameter cannot be 1 for the non-sym TS"))
        else
            RowTransform[dim-1, βIdx[ig-1]] = -1
            RowTransform[dim  , γIdx[ig-1]] = -1

            ColTransform[βIdx[ig-1], dim-1] = -1/2
            ColTransform[γIdx[ig-1], dim ]  = -1/2

            vApproximate[dim-1], vApproximate[dim] = 1/sqrt(2), -sign(bbar)/sqrt(2)
        end
            
    else
        throw(ArgumentError("Only 'symmetric' and 'non_symmetric' values are accepted"))
    end

    basisTransformation = inv(RowTransform)*ColTransform
    _, permMatrix       = permuteHessian(p+1, ig; tsType=tsType)

    vApproximate = (transpose(permMatrix)*basisTransformation)*vApproximate
    normalize!(vApproximate)

    result = Dict();
    result["eigvec_approx"] = vApproximate
    result["eigval_approx"] = -abs(bbar)
    if doChecks
        result["change_basis"] = basisTransformation
        
        ΓTs = transitionState(Γmin, ig, tsType=tsType)
        HTs = hessianCostFunction(qaoa, ΓTs)
        λtrue, ψtrue = eigen(HTs)
        
        innerProd = ψtrue[:,1]' * result["eigvec_approx"]
        result["eigvec_fidelity"] = innerProd
        result["eigval_error"] = abs(result["eigval_approx"]-λtrue[1])/abs(λtrue[1])

        !(abs(innerProd) > 0.95) && println("The obtained approximate eigenvector is wrong. Check!!")
    end
    return result
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
    checks ? @assert(norm(gradCostFunction(qaoa, Γ)) < tol) : nothing

    hessian_matrix = hessianCostFunction(qaoa, Γ)
    return count(x->x<0, eigvals(hessian_matrix))
end