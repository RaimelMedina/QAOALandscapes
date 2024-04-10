# function fourierJacobian(T::Type{<:Real}, p::Int, q::Int=p)
#     coeffmat = zeros(T, p, q)
#     for j ∈ 1:q
#         for i ∈ 1:p
#             coeffmat[i,j] = T((j-1/2)*(i-1/2)*π)
#         end
#     end
#     return coeffmat
# end

# function toFourierParams(Γ::Vector{T}, q::Int=length(Γ)÷2) where T<:Real
#     p = length(Γ) ÷ 2
#     coeffmat = fourierJacobian(T, p, q)
    
#     Γu          = zeros(T, 2q)
#     Γu[1:2:2q] .= ((2/p)*(sin.(coeffmat ./ p)))' * Γ[1:2:2p] .|> T
#     Γu[2:2:2q] .= ((2/p)*(cos.(coeffmat ./ p)))' * Γ[2:2:2p] .|> T
    
#     return Γu
# end

# function fromFourierParams(Γu::Vector{T}, p::Int=length(Γu) ÷ 2) where T<:Real
#     q = length(Γu) ÷ 2
#     coeffmat = fourierJacobian(T, p, q)
    
#     Γnew          = zeros(T, 2p)

#     Γnew[1:2:2p] .= (sin.(coeffmat ./ p)) * Γu[1:2:2q]
#     Γnew[2:2:2p] .= (cos.(coeffmat ./ p)) * Γu[2:2:2q]
    
#     return Γnew
# end

# function fourierInitialization(Γmin::Vector{T}, q::Int=length(Γmin)÷2) where T<:Real
#     p = length(Γmin) ÷ 2
#     if q < p
#         # then we convert to Fourier space and then back to real space but with p+1
#         Γu   = toFourierParams(Γmin, q)
#         Γnew = fromFourierParams(Γu, p+1)
#     else
#         # if q=p (as many frequencies as real-space parameters)
#         # then we add a new frequency and convert back to real space afterwards
#         Γu   = toFourierParams(Γmin, q)
#         append!(Γu, [T(0), T(0)])
#         Γnew = fromFourierParams(Γu)
#     end
#     return Γnew
# end

# function gradCostFunctionFourier(qaoa::QAOA, Γu::Vector{T}, p::Int=length(Γu)÷2) where T<:Real
#     q     = length(Γu) ÷ 2
#     Γ     = fromFourierParams(Γu, p)
#     gradΓ = gradCostFunction(qaoa, Γ)
    
#     coeffmat = fourierJacobian(T, p, q)
    
#     gradΓu          = zeros(T, 2q)
#     gradΓu[1:2:2q] .= (sin.(coeffmat ./ p))' * gradΓ[1:2:2p] .|> T
#     gradΓu[2:2:2q] .= (cos.(coeffmat ./ p))' * gradΓ[2:2:2p] .|> T
#     return gradΓu
# end

# struct FourierInitialization{T <: Real}
#     params::Vector{Vector{T}}
#     R::Int
#     q::Int
# end

# function FourierInitialization(qaoa::QAOA{P, H, M}, vec::Vector{T}, R::Int, q::Int=length(vec)÷2; α=T(0.6)) where {P, H, M, T}
#     @assert R ≥ 0
#     p = length(vec) ÷ 2
#     if R==0
#         return FourierInitialization([fourierInitialization(vec, q)], 0, q)
#     else
#         fvec  = toFourierParams(vec, q)
        
#         # Eq B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
#         variance_vector = fvec .^ 2
#         mat_fvec = zeros(T, length(fvec), R)

#         for i in 1:size(mat_fvec)[1]
#             distrib = Distributions.Normal(T(0), variance_vector[i])

#             # Eq. B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
#             mat_fvec[i, :] = fvec[i] .+ α*rand(T, distrib, R)
#         end

#         for i in 1:R
#             mat_fvec[:, i] .= fromFourierParams(mat_fvec[:, i], p) # increase p here 
#             toFundamentalRegion!(qaoa, view(mat_fvec, :, i)) # fold back parameter into the fundamental region 
#         end

#         new_params = [fourierInitialization(vec, q)]
#         for i in 1:R
#             push!(new_params, fourierInitialization(mat_fvec[:, i], q))
#         end

#         return FourierInitialization{T}(new_params, R, q)
#     end
# end

# function rollDownFourier(qaoa::QAOA{P, H, M}, 
#     Γmin::Vector{T}, 
#     R::Int=0,
#     q::Int=length(Γmin)÷2; 
#     setup=OptSetup(),
#     threaded=false
#     ) where {P, H, M, T}

#     p = length(Γmin) ÷ 2
#     fourierInitData = FourierInitialization(qaoa, Γmin, R, q) # the parameters here are in real space
    
#     if threaded
#         fourierOptimData = ThreadsX.map(
#             x->optimizeParameters(Val(:Fourier), qaoa, toFourierParams(fourierInitData.params[x], q), p+1, setup=setup),
#             1:R+1
#         )
#     else
#         fourierOptimData = map(
#             x->optimizeParameters(Val(:Fourier), qaoa, toFourierParams(fourierInitData.params[x], q), p+1, setup=setup),
#             1:R+1
#         )
#     end
#     E_fourier = [fourierOptimData[x][2] for x in eachindex(fourierOptimData)]
#     Emin_fourier, min_index = findmin(E_fourier)
#     Γmin_fourier = fourierOptimData[min_index][1]
#     return Γmin_fourier, Emin_fourier
# end

# @doc raw"""
#     fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int)
    
# Starting from a local minima `Γ0` at ``p=1`` it performs the `Fourier` optimization strategy until the circuit depth `pmax` is reached.
# By default the `BFGS` optimizer is used. 

# # Arguments 
# * `qaoa::QAOA`: QAOA object 
# * `Γ0::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# # Return
# * `result:Dict`. Dictionary with keys being `keys \in [1, pmax]` and values being a `Tuple{Float64, Vector{Float64}}` of cost function value and corresponding parameter.
# """
# function fourierOptimize(qaoa::QAOA{P, H, M}, 
#     Γ0::Vector{T}, 
#     pmax::Int, 
#     R::Int=0,
#     q::Int=length(Γ0)÷2; 
#     setup=OptSetup(),
#     threaded=false
#     ) where {P, H, M, T}

#     listMinima = Dict{Int, Tuple{T, Vector{T}}}()
#     p = length(Γ0) ÷ 2 
    
#     listMinima[p] = (qaoa(Γ0), Γ0)

#     iter = Progress(pmax-p; desc="Optimizing QAOA energy...")

#     for t ∈ p+1:pmax
#         Γopt, Eopt = rollDownFourier(qaoa, listMinima[t-1][end], R, q, setup=setup, threaded=threaded)
#         listMinima[t] = (Eopt, Γopt)
#         next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
#     end
#     return listMinima
# end

function fourierJacobian(T::Type{<:Real}, p::Int)
    coeffmat = zeros(T, p,p)
    for j ∈ 1:p
        for i ∈ 1:p
            coeffmat[i,j] = T((j-1/2)*(i-1/2)*π)
        end
    end
    return coeffmat
end

function fromFourierParams(Γu::Vector{T}) where T<:Real
    p = length(Γu) ÷ 2
    coeffmat = fourierJacobian(T, p)
    
    Γnew          = zeros(T, 2p)
    Γnew[1:2:2p] .= (sin.(coeffmat ./ p)) * Γu[1:2:2p]
    Γnew[2:2:2p] .= (cos.(coeffmat ./ p)) * Γu[2:2:2p]
    
    return Γnew
end

function toFourierParams(Γ::Vector{T}) where T<:Real
    p = length(Γ) ÷ 2
    coeffmat = fourierJacobian(T, p)
    
    Γu          = zeros(T, 2p)
    Γu[1:2:2p] .= ((2/p)*(sin.(coeffmat ./ p))) * Γ[1:2:2p] .|> T
    Γu[2:2:2p] .= ((2/p)*(cos.(coeffmat ./ p))) * Γ[2:2:2p] .|> T
    
    return Γu
end

function fourierInitialization(Γmin::Vector{T}) where T<:Real
    Γu   = toFourierParams(Γmin)
    append!(Γu, [T(0), T(0)])
    Γnew = fromFourierParams(Γu)
    return Γnew
end

function gradCostFunctionFourier(qaoa::QAOA{P, H, M}, Γu::Vector{T}) where {P, H, M, T<:Real}
    p     = length(Γu) ÷ 2
    Γ     = fromFourierParams(Γu)
    gradΓ = gradCostFunction(qaoa, Γ)
    
    coeffmat = fourierJacobian(T, p)
    
    gradΓu          = zeros(T, 2p)
    gradΓu[1:2:2p] .= (sin.(coeffmat ./ p)) * gradΓ[1:2:2p] .|> T
    gradΓu[2:2:2p] .= (cos.(coeffmat ./ p)) * gradΓ[2:2:2p] .|> T
    return gradΓu
end

struct FourierInitialization{T <: Real}
    params::Vector{Vector{T}}
    R::Int
end

function FourierInitialization(qaoa::QAOA{P, H, M}, vec::Vector{T}, R::Int; α=T(0.6)) where {P, H, M, T<:Real}
    @assert R ≥ 0
    if R==0
        return FourierInitialization([fourierInitialization(vec)], 0)
    else
        fvec  = toFourierParams(vec)
        
        # Eq B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
        variance_vector = fvec .^ 2
        mat_fvec = zeros(T, length(vec), R)

        for i in 1:size(mat_fvec)[1]
            distrib = Distributions.Normal(T(0), variance_vector[i])

            # Eq. B4 from paper https://browse.arxiv.org/pdf/1812.01041.pdf
            mat_fvec[i, :] = fvec[i] .+ α*T.(rand(distrib, R))
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

function rollDownFourier(qaoa::QAOA{P, H, M}, 
    Γmin::Vector{T}, 
    R::Int=0; 
    setup=OptSetup(),
    threaded=false
    ) where {P, H, M, T<:Real}

    fourierInitData = FourierInitialization(qaoa, Γmin, R)
    
    if threaded
        fourierOptimData = ThreadsX.map(
            x->optimizeParameters(Val(:Fourier), qaoa, toFourierParams(fourierInitData.params[x]), setup=setup),
            1:R+1
        )
    else
        fourierOptimData = map(
            x->optimizeParameters(Val(:Fourier), qaoa, toFourierParams(fourierInitData.params[x]), setup=setup),
            1:R+1
        )
    end
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
function fourierOptimize(qaoa::QAOA{P, H, M}, 
    Γ0::Vector{T}, 
    pmax::Int, 
    R::Int=0; 
    setup=OptSetup(),
    threaded=false
    ) where {P, H, M, T<:Real}

    listMinima = Dict{Int, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2 
    
    listMinima[p] = (qaoa(Γ0), Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")

    for t ∈ p+1:pmax
        Γopt, Eopt = rollDownFourier(qaoa, listMinima[t-1][end], R, setup=setup, threaded=threaded)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end
    return listMinima
end