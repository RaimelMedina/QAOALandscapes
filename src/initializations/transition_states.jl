@doc raw"""
    transitionState(Γp::Vector{Float64}, i::Int; tsType='symmetric')

Given an initial state `Γp::Vector{Float64}` of length `2p` it creates another vector
`ΓTs` of size `2p+2` such that the `i`-th γ component of `ΓTs` is 0 and the `i`-th (`i-1`-th) 
β component of `ΓTs` is zero if `tsType='symmetric'` (`tsType='non_symmetric'`) while all the other components
are the same as `Γp`

# Keyword arguments
* `tsType='symmetric'` Only strings values 'symmetric' and 'non_symmetric' are accepted
"""
function transitionState(Γ::AbstractVector{T}, iγ::Int; tsType="symmetric", regularize=false) where T
    p = length(Γ) ÷ 2
    β = Γ[2:2:2p]
    γ = Γ[1:2:2p]

    val = regularize ? cbrt(eps(Float64)) : T(0)
    insert!(γ, iγ, val)
    if tsType == "symmetric"
        insert!(β, iγ, val)
    elseif tsType == "non_symmetric"
        insert!(β, iγ-1, val)
    else
        throw(ArgumentError("Only 'symmetric' and 'non_symmetric' values are accepted"))
    end
    
    ΓTS = zeros(T, 2(p+1))
    ΓTS[2:2:2(p+1)] = β
    ΓTS[1:2:2(p+1)] = γ

    return ΓTS
end

@doc raw"""
    transitionState(Γp::Vector{Float64})

Given an initial state `Γp::Vector{Float64}` of length ``2p`` it creates a matrix ``M_{\rm{TS}}`` of size ``2p+2 \times 2p+1``. 
The columns of ``M_{\rm{TS}}`` corresponds to the transition states associated with the initial minimum `Γp`. The first `p+1` 
columns correspond to symmetric TS while the remaining `p` columns correspond to non-symmetric TS.
"""
function transitionState(Γmin::AbstractVector{T}; regularize=false) where T
    p = length(Γmin) ÷ 2
    vectorOfTS = Vector{T}[]
    
    for i ∈ 1:p+1
        push!(vectorOfTS, transitionState(Γmin, i, tsType="symmetric", regularize=regularize))
    end
    for i ∈ 2:p+1
        push!(vectorOfTS, transitionState(Γmin, i, tsType="non_symmetric", regularize=regularize))
    end
    vectorOfTS = reduce(hcat,vectorOfTS)
end

@doc raw"""
    rollDownTS(qaoa::QAOA, Γmin::Vector{T}, ig::Int; ϵ=0.001, tsType="symmetric")
    
Starting from a local minima we construct a vector corresponding to the transition state specified by `ig`. From there we construct
two new vectors 

```math
\Gamma^0_p = \Gamma_{\rm{TS}} + \epsilon \hat{e}_{\rm{min}},
```

```math
\Gamma^0_m = \Gamma_{\rm{TS}} - \epsilon \hat{e}_{\rm{min}} 
```
We then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee
that the obtained vectors have lower energy than the initial vector `Γmin`

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γmin::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then *roll down* from.
* `ig::Int`: Index of the γ component where the zeros are inserted. 
* `tsType="symmetric"`: In this case, the index of the β component is equal to `ig`. Otherwise, the β index is `ig-1`.
* `optim=Val(:BFGS)`: Means that we will use the L-BFGS algorithm to perform the optimization. The other option is `optim=Val{:GD}`.

# Return
* `result:Tuple`. The returned paramaters are as follows => `Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p`
"""
function rollDownfromTS(qaoa::QAOA, 
    Γmin::AbstractVector{T}, 
    ig::Int; 
    ϵ=0.001, 
    tsType="symmetric", 
    setup=OptSetup()
    ) where T<:Real

    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    umin = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)["eigvec_approx"]
    
    Γ0_p = ΓTs + ϵ*umin
    Γ0_m = ΓTs - ϵ*umin
    
    Γmin_p, Emin_p = optimizeParameters(qaoa, Γ0_p; setup=setup);
    Γmin_m, Emin_m = optimizeParameters(qaoa, Γ0_m; setup=setup);
    
    return [Γmin_m, Γmin_p, [Emin_m, Emin_p]]
end

@doc raw"""
    rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, optim=Val(:BFGS))
    
Starting from a local minima we construct all transition states (a total of 2p+1 of them). From each of the transition states, we construct
two new vectors 

```math
\Gamma^0_p = \Gamma_{\rm{TS}} + \epsilon \hat{e}_{\rm{min}}, 

\Gamma^0_m = \Gamma_{\rm{TS}} - \epsilon \hat{e}_{\rm{min}} 
```

We then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee
that the obtained vectors have lower energy than the initial vector `Γmin`

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γmin::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# Return
* `result:Tuple`. The returned paramaters are as follows => `Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p`
"""
function rollDownTS(qaoa::QAOA, Γmin::Vector{T}; 
    setup=OptSetup(),
    ϵ=0.001, 
    threaded=false
    ) where T<:Real

    p                = length(Γmin) ÷ 2
    parametersResult = Dict{String, Vector{Vector{T}}}()  
    energiesResult   = Dict{String, Vector{T}}()
    dictTS  = Dict{String, Vector{Vector{T}}}()
    
    indices = [[(x, "symmetric") for x in 1:p+1]; [(x, "non_symmetric") for x in 2:p+1]]

    if threaded
        ThreadsX.map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; setup=setup, ϵ=ϵ, tsType=tsType)
        end, indices)
    else
        map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; setup=setup, ϵ=ϵ, tsType=tsType)
        end, indices)
    end
    
    for l in keys(dictTS)
        parametersResult[l] = dictTS[l][1:2]
        energiesResult[l]   = dictTS[l][3]
    end

    return parametersResult, energiesResult
end
# function rollDownTS(qaoa::QAOA, Γmins::Vector{Vector{Float64}}, k::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), ϵ=0.001, threaded=false)
#     params_data = map(x->rollDownTS(qaoa, x, k; method=method, ϵ=ϵ, threaded=threaded), Γmins)
#     params_data = reduce(vcat, params_data)
#     energ_data  = map(x->qaoa(x), params_data)
#     return selectBestParams(energ_data, params_data, k)
# end
