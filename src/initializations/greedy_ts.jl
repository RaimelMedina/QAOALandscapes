function greedySelect(results::Vector{TSResult{T}}) where T
    min_ts = argmin(result -> minimum(result.energies), results)
    # Get index of minimum energy in the winning TSResult
    min_energy_idx = argmin(min_ts.energies)
    # Return both the TSResult and the corresponding parameter column
    return  min_ts.params[:, min_energy_idx], min_ts.energies[min_energy_idx]
end

function optimizeWithGreedy(qaoa::QAOA, 
    Γ0::Vector{T}, 
    pmax::Int,
    init::K,
    alg
    ) where {T<:Real, K<:TSInitialization}

    p = length(Γ0) ÷ 2 
    @assert p < pmax

    energies_optima = zeros(T, pmax-p)
    params_optima = Vector{Vector{T}}(undef, pmax-p)

    params_optima[1] = Γ0
    energies_optima[1] = qaoa(Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")

    for t ∈ 2:(pmax-p)
        results = rollDown(qaoa, params_optima[t-1], init, alg)
        Γopt, Eopt = greedySelect(results)
        
        energies_optima[t] = Eopt
        params_optima[t] = Γopt

        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end
    return energies_optima, params_optima
end

function optimizeWithStrategy(qaoa::QAOA, 
    Γ0::Vector{T}, 
    pmax::Int,
    gamma_index::Int,
    tsType::Val{S},
    init::K,
    alg
    ) where {T<:Real, K<:InterpInitialization, S}

    p = length(Γ0) ÷ 2 
    @assert p < pmax

    energies_optima = zeros(T, pmax-p)
    params_optima= Vector{T}(undef, pmax-p)

    push!(params_optima, Γ0)
    energies_optima[1] = qaoa(Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
    for t ∈ 2:(pmax-p)
        result = rollDown(qaoa, params_optima[t-1], gamma_index, tsType, init, alg)
        min_energy_index = argmin(result.energies)

        energies_optima[t] = result.energies[min_energy_index]
        push!(params_optima, result.params[:, min_energy_index])

        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end
    return Eopt, Γopt
end



# function greedyOptimize(qaoa::QAOA{P, H, M}, 
#     Γ0::Vector{T}, 
#     pmax::Int, 
#     igamma::Int; 
#     tsType="symmetric", 
#     ϵ=T(0.001), 
#     setup=OptSetup()
#     ) where {P, H, M, T<:Real}

#     listMinima = Dict{Int, Tuple{T, Vector{T}}}()
#     p = length(Γ0) ÷ 2
#     listMinima[p] = (qaoa(Γ0), Γ0)

#     iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
#     for t ∈ p+1:pmax
#         dataGreedy = rollDownfromTS(qaoa, listMinima[t-1][end], igamma; ϵ=ϵ, setup=setup, tsType=tsType)
#         Eopt = minimum(dataGreedy[3])
#         Γopt = dataGreedy[findmin(dataGreedy[3])[2]]
#         listMinima[t] = (Eopt, Γopt)
#         next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
#     end

#     return listMinima
# end

# function greedyOptimize(qaoa::QAOA{P, H, M}, Γ0::Vector{T}, 
#     pmax::Int; ϵ=T(0.001), 
#     setup=OptSetup(), 
#     threaded=false
#     ) where {P, H, M, T<:Real}

#     listMinima = Dict{Int, Tuple{T, Vector{T}}}()
#     p = length(Γ0) ÷ 2
#     #Γmin, Emin = optimizeParameters(optim, qaoa, Γ0, method=method)
#     listMinima[p] = (qaoa(Γ0), Γ0)

#     # println("Circuit depth  | Energy    | gradient norm ")
#     # println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
#     iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
#     for t ∈ p+1:pmax
#         Eopt, Γopt = greedySelect(qaoa, listMinima[t-1][end]; ϵ=ϵ, setup=setup, threaded=threaded)
#         listMinima[t] = (Eopt, Γopt)
#         next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
#         # println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
#     end

#     return listMinima
# end