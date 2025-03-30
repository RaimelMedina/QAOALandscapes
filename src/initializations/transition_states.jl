@doc raw"""
    TSInitialization(Γ::Vector{T}, iγ::Int; tsType::Val{S}=Val(:symmetric)) where {T<:Real, S}

Given an initial state `Γ` of length `2p`, creates a vector of length `2p+2` where:
- The `iγ`-th γ component is zero
- If `tsType=Val(:symmetric)`: the `iγ`-th β component is zero
- If `tsType=Val(:non_symmetric)`: the `(iγ-1)`-th β component is zero
All other components remain unchanged from `Γ`.

# Arguments
- `Γ::Vector{T}`: Initial state vector of length 2p
- `iγ::Int`: Index for transition state insertion
- `tsType::Val{S}=Val(:symmetric)`: Transition state type (`:symmetric` or `:non_symmetric`)
"""
function TSInitialization(Γ::Vector{T}, iγ::Int, tsType::Val{S}=Val(:symmetric)) where {T <: Real, S}
    p = length(Γ) ÷ 2
    val = zero(T)
    
    ΓTS = Vector{T}(undef, 2(p+1))
    
    @inbounds for i in 1:iγ-1
        ΓTS[2i-1] = Γ[2i-1]
        ΓTS[2i] = Γ[2i]
    end
    
    ΓTS[2iγ-1] = val
    if S === :symmetric
        ΓTS[2iγ] = val
    elseif S === :non_symmetric
        ΓTS[2(iγ-1)] = val
    else
        throw(ArgumentError("Only :symmetric and :non_symmetric values are accepted"))
    end
    
    @inbounds for i in iγ:p
        ΓTS[2(i+1)-1] = Γ[2i-1]
        ΓTS[2(i+1)] = Γ[2i]
    end
    
    return ΓTS
end

@doc raw"""
    transitionState(Γmin::Vector{T}) where T<:Real

Generates a transition state matrix `MTS` of size `(2p+2) × (2p+1)` from initial state `Γmin`.

# Matrix Structure
- First `p+1` columns: symmetric transition states
- Remaining `p` columns: non-symmetric transition states

# Arguments
- `Γmin::Vector{T}`: Initial minimum state vector of length 2p
"""
function TSInitialization(Γmin::Vector{T}) where T<:Real
    p = length(Γmin) ÷ 2
    vectorOfTS = zeros(T, 2p + 2, 2p + 1)
    
    col = 1
    for i ∈ 1:p+1
        vectorOfTS[:, col] = TSInitialization(Γmin, i, Val(:symmetric))
        col += 1
    end
    for i ∈ 2:p+1
        vectorOfTS[:, col] = TSInitialization(Γmin, i, Val(:non_symmetric))
        col += 1
    end
    return vectorOfTS
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
function rollDown(qaoa::QAOA, 
    Γmin::Vector{T}, 
    ig::Int, 
    tsType::Val{S},
    init::K, alg
    ) where {T<:Real, S, K <: TSInitialization}

    ΓTs = TSInitialization(Γmin, ig, tsType)
    umin = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)["eigvec_approx"] |> Array
    if any(isnan, umin)
        println("WARNING: NaN in umin at ig=$ig, tsType=$tsType")
    end

    Γ0_p = ΓTs + init.ϵ*umin
    Γ0_m = ΓTs - init.ϵ*umin
    
    solutions = [optimizeParameters(qaoa, x, alg) for x in [Γ0_p, Γ0_m]]
    energies = map(x->getproperty(x, :objective), solutions)
    params = map(x->getproperty(x, :u), solutions)

    if any(isnan, energies)
        println("WARNING: NaN in energies at ig=$ig, tsType=$tsType")
    end

    return TSResult(
        reduce(hcat, params), 
        energies,
        ig,
        tsType
    )
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
function rollDown(
        qaoa::QAOA,
        Γmin::Vector{T},
        init::K,
        alg
    ) where {T<:Real, K <: TSInitialization}
    
    p = length(Γmin) ÷ 2
    indices = [[(x, Val(:symmetric)) for x in 1:p+1]; 
    [(x, Val(:non_symmetric)) for x in 2:p+1]]

    optimizationResult = [TSResult{T, indices[i][2] |> typeof}(undef) for i in 1:2p+1]
    # # Create thread-local storage without sizehint
    # thread_results = [TSResult{T}[] for _ in 1:Threads.nthreads()]

    Threads.@threads for v in eachindex(indices)
        idx, ts_type = indices[v]
        result = rollDown(qaoa, Γmin, idx, ts_type, init, alg)
        # Push to thread-local storage
        optimizationResult[v] = result
    end

    # Combine results from all threads
    # optimizationResult = reduce(vcat, thread_results)
    return optimizationResult
end
