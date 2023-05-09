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
    
    ΓTS = zeros(2(p+1))
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
    
    # First, folding β ∈ [-π/4, π/4]
    for i=1:p #beta angles come first, they are between -pi/4, pi/4
        β[i] = mod(β[i], π/2) # folding beta to interval 0, pi/2
        if β[i] > π/4 # translating it to -pi/4, pi/4 interval
            β[i] -= π/2
        end
    end

    graph_degree = degree(qaoa.graph)
    isWeightedG  = typeof(qaoa.graph) <: SimpleWeightedGraph
    if reduce(*, isodd.(graph_degree)) && !isWeightedG 
        # enter here if each vertex has odd degree d. Assuming regular graphs here :|
        # also, this only works for unweighted d-regular random graphs 
        for i=1:p
            γ[i] = mod(γ[i], π) # processing gammas by folding them to -pi/2, pi/2 interval
            if γ[i] > π/2
                γ[i] -= π
            end
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
function rollDownfromTS(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; ϵ=0.001, tsType="symmetric", method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    umin = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)["eigvec_approx"]
    
    Γ0_p = ΓTs + ϵ*umin
    Γ0_m = ΓTs - ϵ*umin
    
    Γmin_p, Emin_p = optimizeParameters(qaoa, Γ0_p; method=method);
    Γmin_m, Emin_m = optimizeParameters(qaoa, Γ0_m; method=method);
    
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
# function rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}, k::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), ϵ=0.001, threaded=false)
#     p                = length(Γmin) ÷ 2
#     parametersResult = Dict{String, Vector{Vector{Float64}}}()  
#     energiesResult   = Dict{String, Vector{Float64}}()
#     dictTS  = Dict{String, Vector{Vector{Float64}}}()
    
#     indices = [[(x, "symmetric") for x in 1:p+1]; [(x, "non_symmetric") for x in 2:p+1]]

#     if threaded
#         ThreadsX.map(pair -> begin
#         x, tsType = pair
#         dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; method=method, ϵ=ϵ, tsType=tsType)
#         end, indices)
#     else
#         map(pair -> begin
#         x, tsType = pair
#         dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; method=method, ϵ=ϵ, tsType=tsType)
#         end, indices)
#     end
    
#     for l in keys(dictTS)
#         parametersResult[l] = dictTS[l][1:2]
#         energiesResult[l]   = dictTS[l][3]
#     end

#     return selectBestParams(energiesResult, parametersResult, k)
# end

function rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), ϵ=0.001, threaded=false)
    p                = length(Γmin) ÷ 2
    parametersResult = Dict{String, Vector{Vector{Float64}}}()  
    energiesResult   = Dict{String, Vector{Float64}}()
    dictTS  = Dict{String, Vector{Vector{Float64}}}()
    
    indices = [[(x, "symmetric") for x in 1:p+1]; [(x, "non_symmetric") for x in 2:p+1]]

    if threaded
        ThreadsX.map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; method=method, ϵ=ϵ, tsType=tsType)
        end, indices)
    else
        map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTS(qaoa, Γmin, x; method=method, ϵ=ϵ, tsType=tsType)
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
