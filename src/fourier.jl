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
    p    = length(Γmin) ÷ 2
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

function fourierJacobian(p::Int64)
    coeffmat = zeros(p,p)
    for j ∈ 1:p
        for i ∈ 1:p
            coeffmat[i,j] = (j-1/2)*(i-1/2)*π
        end
    end
    return coeffmat
end

function rollDownFourier(qaoa::QAOA, Γmin::Vector{Float64})
    ΓFourier = toFourierParams(fourierInitialization(Γmin))

    Γmin_fourier, Emin_fourier = train!(Val(:Fourier), qaoa, ΓFourier; printout = false);
    return Γmin_fourier, Emin_fourier
end


function fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int)
    listMinima = Dict{Int64, Tuple{Float64, Vector{Float64}}}()
    p = length(Γ0) ÷ 2 
    Γmin, Emin = train!(Val(:BFGS), qaoa, Γ0; printout = false)
    listMinima[p] = (Emin, Γmin)

    println("Circuit depth  | Energy    | gradient norm ")
    println("    p=$(p)     | $(round(Emin, digits = 7)) | $(norm(gradCostFunction(qaoa, Γmin)))")

    for t = p+1:pmax
        Γopt, Eopt = rollDownFourier(qaoa, listMinima[t-1][end])
        listMinima[t] = (Eopt, Γopt)
        
        println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end
    return listMinima
end