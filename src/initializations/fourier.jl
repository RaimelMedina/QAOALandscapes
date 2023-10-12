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
    Γu[1:2:2p] .= ((2/p)*(sin.(coeffmat ./ p))) * Γ[1:2:2p]
    Γu[2:2:2p] .= ((2/p)*(cos.(coeffmat ./ p))) * Γ[2:2:2p]
    
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

struct FourierInitialization{T <: Real}
    params::Vector{Vector{T}}
    R::Int
end

function FourierInitialization(qaoa::QAOA, vec::Vector{T}, R::Int; α=0.6) where T<:Real
    @assert R ≥ 0
    if R==0
        return FourierInitialization([fourierInitialization(vec)], 0)
    else
        fvec  = toFourierParams(vec)
        
        # Eq B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
        variance_vector = fvec .^ 2
        mat_fvec = zeros(length(vec), R)

        for i in 1:size(mat_fvec)[1]
            distrib = Distributions.Normal(0.0, variance_vector[i])

            # Eq. B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
            mat_fvec[i, :] = fvec[i] .+ α*rand(distrib, R)
        end

        for i in 1:R
            mat_fvec[:, i] .= fromFourierParams(mat_fvec[:, i])
            toFundamentalRegion!(qaoa, view(mat_fvec, :, i)) 
        end

        new_params = [fourierInitialization(vec)]
        for i in 1:R
            push!(new_params, fourierInitialization(mat_fvec[:, i]))
        end

        return FourierInitialization{T}(new_params, R)
    end
end

function rollDownFourier(qaoa::QAOA, 
    Γmin::Vector{T}, 
    R::Int=0; 
    setup=OptSetup()
    ) where T<:Real

    fourierInitData = FourierInitialization(qaoa, Γmin, R)
    
    fourierOptimData = ThreadsX.map(
        x->optimizeParameters(Val(:Fourier), qaoa, toFourierParams(fourierInitData.params[x]), setup=setup),
        1:R+1
    )
    E_fourier = [fourierOptimData[x][2] for x in eachindex(fourierOptimData)]
    Emin_fourier, min_index = findmin(E_fourier)
    Γmin_fourier = fourierOptimData[min_index][1]
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
function fourierOptimize(qaoa::QAOA, 
    Γ0::Vector{T}, 
    pmax::Int, 
    R::Int=0; 
    setup=OptSetup()
    ) where T<:Real

    listMinima = Dict{Int64, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2 
    #Γmin, Emin = optimizeParameters(qaoa, Γ0; settings=settings)
    listMinima[p] = (qaoa(Γ0), Γ0)

    # println("Circuit depth  | Energy    | gradient norm ")
    # println("    p=$(p)     | $(round(listMinima[p][1], digits = 7)) | $(norm(gradCostFunction(qaoa, listMinima[p][2])))")
    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")

    for t ∈ p+1:pmax
        Γopt, Eopt = rollDownFourier(qaoa, listMinima[t-1][end], R, setup=setup)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
        #println("    p=$(t)     | $(round(Eopt, digits = 7)) | $(norm(gradCostFunction(qaoa, Γopt)))")
    end
    return listMinima
end