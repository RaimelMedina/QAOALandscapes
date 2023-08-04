function fourierJacobian(p::Int)
    coeffmat = zeros(p,p)
    for j ∈ 1:p
        for i ∈ 1:p
            coeffmat[i,j] = (j-1/2)*(i-1/2)*π
        end
    end
    return coeffmat
end

function fromFourierParams(Γu::Vector{Float64})
    p = length(Γu) ÷ 2
    coeffmat = fourierJacobian(p)
    
    Γnew          = zeros(2p)
    Γnew[1:2:2p] .= (sin.(coeffmat ./ p)) * Γu[1:2:2p]
    Γnew[2:2:2p] .= (cos.(coeffmat ./ p)) * Γu[2:2:2p]
    
    return Γnew
end

function toFourierParams(Γ::Vector{Float64})
    p = length(Γ) ÷ 2
    coeffmat = fourierJacobian(p)
    
    Γu          = zeros(2p)
    Γu[1:2:2p] .= (sin.(coeffmat ./ p)) \ Γ[1:2:2p]
    Γu[2:2:2p] .= (cos.(coeffmat ./ p)) \ Γ[2:2:2p]
    
    return Γu
end

function fourierInitialization(Γmin::Vector{Float64})
    Γu   = toFourierParams(Γmin)
    append!(Γu, [0.0, 0.0])
    Γnew = fromFourierParams(Γu)
    return Γnew
end

function gradCostFunctionFourier(qaoa::QAOA, Γu::Vector{Float64})
    p     = length(Γu) ÷ 2
    Γ     = fromFourierParams(Γu)
    gradΓ = gradCostFunction(qaoa, Γ)
    
    coeffmat = fourierJacobian(p)
    
    gradΓu          = zeros(2p)
    gradΓu[1:2:2p] .= (sin.(coeffmat ./ p)) * gradΓ[1:2:2p]
    gradΓu[2:2:2p] .= (cos.(coeffmat ./ p)) * gradΓ[2:2:2p]
    return gradΓu
end

function rollDownFourier(qaoa::QAOA, Γmin::Vector{Float64}; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    ΓFourier = toFourierParams(fourierInitialization(Γmin))
    Γmin_fourier, Emin_fourier = optimizeParameters(Val(:Fourier), qaoa, ΓFourier, method=method)
    return Γmin_fourier, Emin_fourier
end

@doc raw"""
    fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int)
    
Starting from a local minima `Γ0` at ``p=1`` it performs the `Fourier` optimization strategy until the circuit depth `pmax` is reached.
By default the `BFGS` optimizer is used. 

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γ0::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# Return
* `result:Dict`. Dictionary with keys being `keys \in [1, pmax]` and values being a `Tuple{Float64, Vector{Float64}}` of cost function value and corresponding parameter.
"""
function fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = length(Γ0) ÷ 2 
    #Γmin, Emin = optimizeParameters(qaoa, Γ0; settings=settings)
    listMinima[p] = (qaoa(Γ0), Γ0)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")

    for t = p+1:pmax
        Γopt, Eopt = rollDownFourier(qaoa, listMinima[t-1][end], method=method)
        listMinima[t] = (Eopt, Γopt)
        
        println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end
    return listMinima
end