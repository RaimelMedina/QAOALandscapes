@doc raw"""
    transitionState(Γp::Vector{Float64}, i::Int; tsType='symmetric')

Given an initial state `Γp::Vector{Float64}` of length `2p` it creates another vector
`ΓTs` of size `2p+2` such that the `i`-th γ component of `ΓTs` is 0 and the `i`-th (`i-1`-th) 
β component of `ΓTs` is zero if `tsType='symmetric'` (`tsType='non_symmetric'`) while all the other components
are the same as `Γp`

# Keyword arguments
* `tsType='symmetric'` Only strings values 'symmetric' and 'non_symmetric' are accepted
"""
function transitionState(Γ::Vector{Float64}, iγ::Int; tsType="symmetric")
    p = length(Γ) ÷ 2
    β = Γ[2:2:2p]
    γ = Γ[1:2:2p]

    insert!(γ, iγ, 0.0)
    if tsType == "symmetric"
        insert!(β, iγ, 0.0)
    elseif tsType == "non_symmetric"
        insert!(β, iγ-1, 0.0)
    else
        throw(ArgumentError("Only 'symmetric' and 'non_symmetric' values are accepted"))
    end
    
    ΓTS = zeros(2(p+1))
    ΓTS[2:2:2(p+1)] = β
    ΓTS[1:2:2(p+1)] = γ

    return ΓTS
end

@doc raw"""
    toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})

Implements the symmetries of the QAOA for different graphs. For more detail see the following [`reference`](https://arxiv.org/abs/2209.01159).

For an arbitrary graph, we can restrict both ``\gamma`` and ``\beta`` parameters to the ``[-\pi/2, \pi/2]`` interval. Furthermore, ``\beta`` parameters
can be restricted even further to the ``[-\pi/4, \pi/4]`` interval (see [`here`](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021067))
Finally, when dealing with regular graphs with odd degree `\gamma` paramaters can be brought to the ``[-\pi/4, \pi/4]`` interval.
This function modifies inplace the initial input vector ``Γ``. 
"""
function toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})
    p = length(Γ) ÷ 2
    β = Γ[2:2:2p]
    γ = Γ[1:2:2p]
    
    # First, folding β ∈ [-π/4, π/4] and
    # γ ∈ [-π/2, π/2]
    for i=1:p #beta angles come first, they are between -pi/4, pi/4
        β[i] = mod(β[i], π/2) # folding beta to interval 0, pi/2
        if β[i] > π/4 # translating it to -pi/4, pi/4 interval
            β[i] -= π/2
        end
        γ[i] = mod(γ[i], π) # processing gammas by folding them to -pi/2, pi/2 interval
        if γ[i] > π/2
            γ[i] -= π
        end
    end

    graph_degree = degree(qaoa.graph)
    if reduce(*, isodd.(graph_degree)) # enter here if each vertex has odd degree. Assuming regular graphs here :| 
        for i=1:p
            # γ[i] = mod(γ[i], π) # processing gammas by folding them to -pi/2, pi/2 interval
            # if γ[i] > π/2
            #     γ[i] -= π
            # end
            if abs(γ[i]) > π/4 # now folding them even more: to -pi/4, pi/4 interval
                β[i:end] .*= -1 # this requires sign flip of betas!
                γ[i] -= sign(γ[i])*π/2
            end
            if γ[1] < 0 # making angle gamma_1 positive
                β .*= -1  # by changing the sign of ALL angles
                γ .*= -1
            end
        end
    end
    # elseif isdRegularGraph(qaoa.graph, 2)
    #     for i=1:p
    #         γ[i] = mod(γ[i], π) # processing gammas by folding them to -pi/2, pi/2 interval
    #         if γ[i] > π/2
    #             γ[i] -= π
    #         end
    #         if abs(γ[i]) > π/4 # now folding them even more: to -pi/4, pi/4 interval
    #             β[i:end] .*= -1 # this requires sign flip of betas!
    #             γ[i] -= sign(γ[i])*π/2
    #         end
    #         if γ[1] < 0 # making angle gamma_1 positive
    #             β .*= -1  # by changing the sign of ALL angles
    #             γ .*= -1
    #         end
    #     end
    # else

    Γ[2:2:2p] = β;
    Γ[1:2:2p] = γ;
end

@doc raw"""
    rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; ϵ=0.001, tsType="symmetric")
    
Starting from a local minima we construct a vector corresponding to the transition state specified by `ig`. From there we construct
two new vectors 

```math
\Gamma^0_p = \Gamma_{\rm{TS}} + \epsilon \hat{e}_{\rm{min}},

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
function rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; ϵ=0.001, tsType="symmetric", optim=Val(:BFGS))
    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    result = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)
    
    Γ0_p = ΓTs + ϵ*result["eigvec_approx"]
    Γ0_m = ΓTs - ϵ*result["eigvec_approx"]
    
    Γmin_p, Emin_p = optimizeParameters(optim, qaoa, Γ0_p; printout = false);
    Γmin_m, Emin_m = optimizeParameters(optim, qaoa, Γ0_m; printout = false);
    
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
function rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, optim=Val(:BFGS), threaded=false, chooseSmooth=false)
    p                = length(Γmin) ÷ 2
    parametersResult = Dict{String, Vector{Vector{Float64}}}()  
    energiesResult   = Dict{String, Vector{Float64}}()
    dictSymmetricTS  = Dict{String, Vector{Vector{Float64}}}()
    
    if threaded
        ThreadsX.map(x->dictSymmetricTS[string((x,x))]   = rollDownTS(qaoa, Γmin, x; ϵ=ϵ, tsType="symmetric", optim=optim), 1:p+1)
        ThreadsX.map(x->dictSymmetricTS[string((x,x-1))] = rollDownTS(qaoa, Γmin, x; ϵ=ϵ, tsType="non_symmetric", optim=optim), 2:p+1)
    else
        for x in 1:p+1
            setindex!(dictSymmetricTS, rollDownTS(qaoa, Γmin, x; ϵ=ϵ, tsType="symmetric", optim=optim), string((x,x)))
            
            if x != 1
                setindex!(dictSymmetricTS, rollDownTS(qaoa, Γmin, x; ϵ=ϵ, tsType="non_symmetric", optim=optim), string((x,x-1)))
            end
        end
    end
    
    for k in keys(dictSymmetricTS)
        parametersResult[k] = dictSymmetricTS[k][1:2]
        energiesResult[k]   = dictSymmetricTS[k][3]
    end
    
    if (chooseSmooth && p>3)
        for k in keys(dictSymmetricTS)
            smoothParams = selectSmoothParameter(parametersResult[k])
            parametersResult[k] = [smoothParams[2]]
            energiesResult[k]   = [energiesResult[k][smoothParams[1]]]
        end
    end
    return parametersResult, energiesResult
end