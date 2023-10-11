@doc raw"""
    getNegativeHessianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric")

Computes the approximation to the minimum (negative) eigenvalue of the Hessian at the TS obtained by padding with zeros
the local minimum `Γmin`. The transition state is completely specified by the index of the γ component `ig`, and the 
type of transition states (`"symmetric"` or `"non_symmetric"`). The cost of obtaining this approximate eigenvalue is basically
the cost of computing two matrix elements of a Hessian.
"""
function getNegativeHessianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType="symmetric")
    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    p    = length(Γmin) ÷ 2
    
    γIdx = 1:2:(2p+2)
    βIdx = 2:2:(2p+2)

    b    = 0.0
    bbar = 0.0

    if tsType=="symmetric"
        b = hessianCostFunction(qaoa, ΓTs, [γIdx[ig], βIdx[ig]])
        if (ig != 1 && ig != p+1)
            bbar = b - hessianCostFunction(qaoa, Γmin, [γIdx[ig], βIdx[ig-1]])
            bbar /= 2
        else      
            bbar = b/sqrt(2)
        end
    elseif tsType=="non_symmetric"
        b = hessianCostFunction(qaoa, ΓTs, [γIdx[ig], βIdx[ig-1]])
        bbar = b - hessianCostFunction(qaoa, Γmin, [γIdx[ig-1], βIdx[ig-1]])
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
function permuteHessian(H::AbstractArray{Float64,2}, i::Int; tsType="symmetric")
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

