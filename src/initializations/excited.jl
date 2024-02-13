function Hzz2_ψ!(qaoa::QAOA{T1, T2}, psi::Vector{Complex{T2}}, Etarget::T2) where {T1<:AbstractGraph, T2<:Real}
    Threads.@threads for i in eachindex(qaoa.HC, psi)
        psi[i] *= (qaoa.HC[i] - Etarget)^2
    end
    return nothing
end


function gradient_excited!(G::Vector{T2}, qaoa::QAOA{T1, T2, T3}, gradTape::GradientTape{T2}, params::Vector{T2}, Etarget::T2) where {T1 <:AbstractGraph, T2<: Real, T3<:AbstractBackend}
    # this will update/populate qaoa.state which we will call |λ⟩ following the paper
    gradTape.λ = getQAOAState(qaoa, params)
    # |ϕ⟩ := |λ⟩
    gradTape.ϕ .= gradTape.λ
    
    # needed to not allocate a new array when doing Hx|ψ⟩
    gradTape.ξ .= gradTape.λ
    
    # |λ⟩ := H |λ⟩
    Hzz2_ψ!(qaoa, gradTape.λ, Etarget)
    
    # now we allocate |μ⟩
    # μ = similar(λ)

    for i in length(params):-1:1
        # |ϕ⟩ ← (Uᵢ)†|ϕ⟩    
        applyQAOALayer!(qaoa, -params[i], i, gradTape.ϕ)
        
        # |μ⟩ ← |ϕ⟩
        gradTape.μ .= gradTape.ϕ

        # |μ⟩ ← dUᵢ/dθᵢ |μ⟩
        applyQAOALayerDerivative!(qaoa, params[i], i, gradTape.μ, gradTape.ξ)
        
        # ∇Eᵢ = 2 ℜ ⟨ λ | μ ⟩
        G[i] = T2(2)*real(dot(gradTape.λ, gradTape.μ))
        if i > 1
            #|λ⟩ ← (Uᵢ)†|λ⟩
            applyQAOALayer!(qaoa, -params[i], i, gradTape.λ)
        end
    end 
    return nothing
end

function gradCostFunctionExcited(qaoa::QAOA{T1, T2, T3}, params::Vector{T2}, Etarget::T2) where {T1 <:AbstractGraph, T2<: Real, T3<:AbstractBackend}
    # this will update/populate qaoa.state which we will call |λ⟩ following the paper
    λ = getQAOAState(qaoa, params) # U(Γ) |+⟩
    κ = copy(λ)
    # |ϕ⟩ := |λ⟩
    ϕ = copy(λ)

    # |λ⟩ := H |λ⟩
    Hzz2_ψ!(qaoa, λ, Etarget)
    
    # now we allocate |μ⟩
    μ = similar(λ)

    gradResult = zeros(T2, length(params))
    
    for i in length(params):-1:1
        # |ϕ⟩ ← (Uᵢ)†|ϕ⟩    
        applyQAOALayer!(qaoa, -params[i], i, ϕ)
        
        # |μ⟩ ← |ϕ⟩
        μ .= ϕ

        # |μ⟩ ← dUᵢ/dθᵢ |μ⟩
        applyQAOALayerDerivative!(qaoa, params[i], i, μ, κ)
        # applyQAOALayer!(qaoa, params[i], i, μ)
        # if isodd(i)
        #     Hzz_ψ!(qaoa, μ)
        # else
        #     Hx_ψ!(qaoa, μ)
        # end
        # μ *= -1.0*im
        
        # ∇Eᵢ = 2 ℜ ⟨ λ | μ ⟩
        gradResult[i] = T2(2)*real(dot(λ, μ))
        if i > 1
            #|λ⟩ ← (Uᵢ)†|λ⟩
            applyQAOALayer!(qaoa, -params[i], i, λ)
        end
    end 
    return gradResult
end

function optimizeParametersExcited( 
    qaoa::QAOA{T1, T, T3}, 
    params, 
    Etarget;
    linesearch=Optim.BackTracking(order=3)
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    f(x) = qaoa(x, Etarget)
    grd(x) = 0.5*norm(gradCostFunction(qaoa, x))

    gradTape = GradientTape(qaoa)
    
    function g!(G,x)
        gradient_excited!(G, qaoa, gradTape, x, Etarget)
        # G .*= 2*sqrt(f(x))
        # G .+= FiniteDiff.FiniteDiff.finite_difference_gradient(grd, x)
    end
    # function ∇f!(G, x)
    #     G .= ForwardDiff.gradient(s->f(s), x)
    # end

    result = Optim.optimize(f, g!, params, method=Optim.BFGS(linesearch=linesearch))

    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)
    convergence_info = Optim.converged(result)
    toFundamentalRegion!(qaoa, parameters)

    if !convergence_info
        if f(params) < cost
            throw(AssertionError("Optimization did not converged. Final gradient norm is |∇E|=$(norm(gradCostFunction(qaoa, parameters)))"))
        else
            println("Optimization did not converged but energy decreased")
        end
    end

    return parameters, cost
end


function rollDownfromTSExcited(qaoa::QAOA{T1, T, T3}, 
    Γmin, 
    ig::Int,
    Etarget; 
    ϵ=T(0.001),
    linesearch=Optim.BackTracking(order=3), 
    tsType="symmetric"
    ) where {T1 <:AbstractGraph, T<:Real, T3<:AbstractBackend}

    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    umin = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)["eigvec_approx"] |> Array
    
    Γ0_p = ΓTs + ϵ*umin
    Γ0_m = ΓTs - ϵ*umin
    
    Γmin_p, Emin_p = optimizeParametersExcited(qaoa, Γ0_p, Etarget; linesearch=linesearch)
    Γmin_m, Emin_m = optimizeParametersExcited(qaoa, Γ0_m, Etarget; linesearch=linesearch)
    
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
function rollDownTSExcited(qaoa::QAOA{T1, T, T3}, Γmin, Etarget;
    linesearch=Optim.BackTracking(order=3), 
    ϵ=T(0.001), 
    threaded=false
    ) where {T1 <:AbstractGraph, T<:Real, T3<:AbstractBackend}

    p                = length(Γmin) ÷ 2
    parametersResult = Dict{String, Vector{Vector{T}}}()  
    energiesResult   = Dict{String, Vector{T}}()
    dictTS  = Dict{String, Vector{Vector{T}}}()
    
    indices = [[(x, "symmetric") for x in 1:p+1]; [(x, "non_symmetric") for x in 2:p+1]]

    if threaded
        ThreadsX.map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTSExcited(qaoa, Γmin, x, Etarget; ϵ=ϵ, tsType=tsType, linesearch=linesearch)
        end, indices)
    else
        map(pair -> begin
        x, tsType = pair
        dictTS[string((x, x - (tsType == "non_symmetric")))] = rollDownfromTSExcited(qaoa, Γmin, x, Etarget; ϵ=ϵ, tsType=tsType, linesearch=linesearch)
        end, indices)
    end
    
    for l in keys(dictTS)
        parametersResult[l] = dictTS[l][1:2]
        energiesResult[l]   = dictTS[l][3]
    end

    return parametersResult, energiesResult
end

function greedySelectExcited(qaoa::QAOA{T1, T, T3}, 
    Γmin::Vector{T},
    Etarget; 
    ϵ=T(0.001),
    linesearch=Optim.BackTracking(order=3),
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    paramResult, energyResult = rollDownTSExcited(qaoa, Γmin, Etarget; ϵ=ϵ, threaded=threaded, linesearch=linesearch)
    # get key of minimum energy #
    valMinimum, keyMinimum = findmin(energyResult);
    # determine which point to take #
    minIdx = argmin(valMinimum)

    return valMinimum[minIdx], paramResult[keyMinimum][minIdx]
end

function greedyOptimizeExcited(qaoa::QAOA{T1, T, T3}, Γ0::Vector{T}, Etarget,
    pmax::Int; 
    ϵ=T(0.001),
    linesearch=Optim.BackTracking(order=3), 
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2
    #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
    listMinima[p] = ((qaoa(Γ0)-Etarget)^2, Γ0)

    # println("Circuit depth  | Energy    | gradient norm ")
    # println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    for t ∈ p+1:pmax
        Eopt, Γopt = greedySelectExcited(qaoa, listMinima[t-1][end], Etarget; ϵ=ϵ, threaded=threaded, linesearch=linesearch)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
        # println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end

function greedyOptimizeExcited(qaoa::QAOA{T1, T, T3}, 
    Γ0::Vector{T}, 
    Etarget,
    pmax::Int, 
    igamma::Int; 
    linesearch=Optim.BackTracking(order=3),
    tsType="symmetric", 
    ϵ=T(0.001)
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2
    listMinima[p] = ((qaoa(Γ0) - Etarget)^2, Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
    for t ∈ p+1:pmax
        dataGreedy = rollDownfromTSExcited(qaoa, listMinima[t-1][end], igamma, Etarget; ϵ=ϵ, tsType=tsType, linesearch=linesearch)
        Eopt = minimum(dataGreedy[3])
        Γopt = dataGreedy[findmin(dataGreedy[3])[2]]
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end

    return listMinima
end