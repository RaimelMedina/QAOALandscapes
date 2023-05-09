function greedySelect(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), threaded=false, chooseSmooth=false)
    paramResult, energyResult = rollDownTS(qaoa, Γmin; ϵ=ϵ, method=method, threaded=threaded, chooseSmooth=chooseSmooth)
    # get key of minimum energy #
    valMinimum, keyMinimum = findmin(energyResult);
    # determine which point to take #
    minIdx = argmin(valMinimum)

    return valMinimum[minIdx], paramResult[keyMinimum][minIdx]
end

function greedyOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int, igamma::Int; tsType="symmetric", ϵ=0.001, method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), chooseSmooth=false)
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = length(Γ0) ÷ 2
    #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
    listMinima[p] = (qaoa(Γ0), Γ0)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    
    for t ∈ p+1:pmax
        dataGreedy = rollDownTS(qaoa, listMinima[t-1][end], igamma; ϵ=ϵ, method=method, tsType=tsType)
        if chooseSmooth
            idxSmooth, paramSmooth = selectSmoothParameter(dataGreedy[1], dataGreedy[2])
            Eopt = dataGreedy[3][idxSmooth]
            Γopt = paramSmooth
        else
            Eopt = minimum(dataGreedy[3])
            Γopt = dataGreedy[findmin(dataGreedy[3])[2]]
        end
        listMinima[t] = (Eopt, Γopt)
        println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end

function greedyOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; ϵ=0.001, method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), threaded=false, chooseSmooth=false)
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = length(Γ0) ÷ 2
    #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
    listMinima[p] = (qaoa(Γ0), Γ0)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    
    for t ∈ p+1:pmax
        Eopt, Γopt = greedySelect(qaoa, listMinima[t-1][end]; ϵ=ϵ, method=method, threaded=threaded, chooseSmooth=chooseSmooth)
        listMinima[t] = (Eopt, Γopt)
        println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end

    return listMinima
end