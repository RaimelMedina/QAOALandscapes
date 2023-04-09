function greedySelect(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, optim=Val(:BFGS), threaded=false, chooseSmooth=false)
    paramResult, energyResult = rollDownTS(qaoa, Γmin; ϵ=ϵ, optim=optim, threaded=threaded, chooseSmooth=chooseSmooth)
    # get key of minimum energy #
    valMinimum, keyMinimum = findmin(energyResult);
    # determine which point to take #
    minIdx = argmin(valMinimum)

    return valMinimum[minIdx], paramResult[keyMinimum][minIdx]
end

function greedyOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int, igamma::Int; tsType="symmetric", ϵ=0.001, optim=Val(:BFGS), chooseSmooth=false)
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = 1
    Γmin, Emin = optimizeParameters(optim, qaoa, Γ0; printout = false)
    listMinima[p] = (Emin, Γmin)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(Emin, digits = 7)) | $(norm(gradCostFunction(qaoa, Γmin)))")
    
    for p ∈ 2:pmax
        dataGreedy = rollDownTS(qaoa, listMinima[p-1][end], igamma; ϵ=ϵ, optim=optim, tsType=tsType)
        if chooseSmooth
            idxSmooth, paramSmooth = selectSmoothParameter(dataGreedy[1], dataGreedy[2])
            Eopt = dataGreedy[3][idxSmooth]
            Γopt = paramSmooth
        else
            Eopt = minimum(dataGreedy[3])
            Γopt = dataGreedy[findmin(dataGreedy[3])[2]]
        end
        listMinima[p] = (Eopt, Γopt)
        println("    p=$(p)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end

function greedyOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; ϵ=0.001, optim=Val(:BFGS), threaded=false, chooseSmooth=false)
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = 1
    Γmin, Emin = optimizeParameters(optim, qaoa, Γ0; printout = false)
    listMinima[p] = (Emin, Γmin)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(Emin, digits = 7)) | $(norm(gradCostFunction(qaoa, Γmin)))")
    
    for p ∈ 2:pmax
        Eopt, Γopt = greedySelect(qaoa, listMinima[p-1][end]; ϵ=ϵ, optim=optim, threaded=threaded, chooseSmooth=chooseSmooth)
        listMinima[p] = (Eopt, Γopt)
        println("    p=$(p)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end