function greedySelect(qaoa::QAOA{T1, T, T3}, 
    Γmin::Vector{T}; 
    ϵ=T(0.001), 
    setup=OptSetup(), 
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    paramResult, energyResult = rollDownTS(qaoa, Γmin; ϵ=ϵ, setup=setup, threaded=threaded)
    # get key of minimum energy #
    valMinimum, keyMinimum = findmin(energyResult);
    # determine which point to take #
    minIdx = argmin(valMinimum)

    return valMinimum[minIdx], paramResult[keyMinimum][minIdx]
end

function greedySelectFidelity(qaoa::QAOA{T1, T, T3}, 
    Γmin::Vector{T},
    state_index::Int; 
    ϵ=T(0.001),
    setup=OptSetup(), 
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    paramResult, _ = rollDownTS(qaoa, Γmin; ϵ=ϵ, setup=setup, threaded=threaded)
    
    vec_of_params = vcat(values(paramResult)...)
    vec_of_fidelities = ThreadsX.map(x->
        abs2(getindex(
            getQAOAState(qaoa, x), state_index
        )),
        vec_of_params
    )
    
    (fid_val, fid_idx) = findmax(vec_of_fidelities)

    println("Optimal fidelity with excited state i=$(state_index) is f=$(fid_val)")
    return qaoa(vec_of_params[fid_idx]), vec_of_params[fid_idx]
end

function greedyOptimize(qaoa::QAOA{T1, T, T3}, 
    Γ0::Vector{T}, 
    pmax::Int, 
    igamma::Int; 
    tsType="symmetric", 
    ϵ=T(0.001), 
    setup=OptSetup()
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2
    listMinima[p] = (qaoa(Γ0), Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
    for t ∈ p+1:pmax
        dataGreedy = rollDownfromTS(qaoa, listMinima[t-1][end], igamma; ϵ=ϵ, setup=setup, tsType=tsType)
        Eopt = minimum(dataGreedy[3])
        Γopt = dataGreedy[findmin(dataGreedy[3])[2]]
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end

    return listMinima
end

function greedyOptimize(qaoa::QAOA{T1, T, T3}, Γ0::Vector{T}, 
    pmax::Int; ϵ=T(0.001), 
    setup=OptSetup(), 
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2
    #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
    listMinima[p] = (qaoa(Γ0), Γ0)

    # println("Circuit depth  | Energy    | gradient norm ")
    # println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    for t ∈ p+1:pmax
        Eopt, Γopt = greedySelect(qaoa, listMinima[t-1][end]; ϵ=ϵ, setup=setup, threaded=threaded)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
        # println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end

function greedyOptimizeFidelity(qaoa::QAOA{T1, T, T3}, Γ0::Vector{T}, 
    pmax::Int,
    state_index; 
    ϵ=T(0.001), 
    setup=OptSetup(), 
    threaded=false
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2
    #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
    listMinima[p] = (qaoa(Γ0), Γ0)

    # println("Circuit depth  | Energy    | gradient norm ")
    # println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    for t ∈ p+1:pmax
        Eopt, Γopt = greedySelectFidelity(qaoa, listMinima[t-1][end], state_index; ϵ=ϵ, setup=setup, threaded=threaded)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
        # println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end