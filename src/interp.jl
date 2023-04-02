@doc raw"""
    interpInitialization(Γp::Vector{Float64})

Given an initial state `Γp::Vector{Float64}` of length `2p` it creates another vector
`ΓInterp` of size `2p+2` with γ (β) components given by the following expression \\

``
\gamma^i_{p+1} = \frac{i-1}{p} \gamma^{i-1}_{p} + \frac{p-i+1}{p}\gamma^{i}_{p}
``
"""
function interpInitialization(Γ::Vector{Float64})
    p = length(Γ) ÷ 2
    β = Γ[2:2:2p]
    γ = Γ[1:2:2p]
    
    βNew = map(x->((x-1)/p)*(x==1 ? 0 : β[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : β[x]), 1:p+1)
    γNew = map(x->((x-1)/p)*(x==1 ? 0 : γ[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : γ[x]), 1:p+1)

    ΓNew = zeros(2*(p+1))
    ΓNew[2:2:2(p+1)] = βNew
    ΓNew[1:2:2(p+1)] = γNew

    return ΓNew
end

@doc raw"""
    rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; optim = Val(:BFGS))
    
Starting from a local minima we construct a new vector using the INTERP initialization. From there we carry out the optimization
algorithm using the `optim=Val(:BFGS)` optimizer (otherwise `optim=Val(:GD)`) 

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γmin::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# Return
* `result:Tuple`. The first element corresponds to the vector corresponding to which the algorithm converged to, and the second element is correponding energy_history
"""
function rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; optim = Val(:BFGS))
    ΓInterp = interpInitialization(Γmin)

    Γmin_interp, Emin_interp = train!(optim, qaoa, ΓInterp; printout = false);
    return Γmin_interp, Emin_interp
end

function interpOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; optim = Val(:BFGS))
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = length(Γ0) ÷ 2 
    Γmin, Emin = train!(optim, qaoa, Γ0; printout = false)
    listMinima[p] = (Emin, Γmin)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(Emin, digits = 7)) | $(norm(gradCostFunction(qaoa, Γmin)))")

    for t = p+1:pmax
        Γopt, Eopt = rollDownInterp(qaoa, listMinima[t-1][end]; optim = optim)
        listMinima[t] = (Eopt, Γopt)
        
        println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end
    return listMinima
end