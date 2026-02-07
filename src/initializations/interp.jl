function (init::InterpInitialization)(Γ::Vector{T}) where T<:Real
    p = length(Γ) ÷ 2
    β = @view Γ[2:2:2p]
    γ = @view Γ[1:2:2p]

    βNew = Vector{T}(undef, p + 1)
    γNew = Vector{T}(undef, p + 1)

    @inbounds for x in 1:(p + 1)
        left_weight = (x - 1) / p
        right_weight = (p - x + 1) / p
        βNew[x] = left_weight * (x == 1 ? zero(T) : β[x - 1]) +
                  right_weight * (x == p + 1 ? zero(T) : β[x])
        γNew[x] = left_weight * (x == 1 ? zero(T) : γ[x - 1]) +
                  right_weight * (x == p + 1 ? zero(T) : γ[x])
    end

    ΓNew = Vector{T}(undef, 2 * (p + 1))
    @inbounds for x in 1:(p + 1)
        ΓNew[2x] = βNew[x]
        ΓNew[2x - 1] = γNew[x]
    end

    return ΓNew
end

function rollDown(qaoa::QAOA, Γ::Vector{T}, init::K, alg)::Tuple{Vector{T}, T} where {K<:InterpInitialization, T<:Real}
    Γinterp = init(Γ)
    sol = optimizeParameters(qaoa, Γinterp, alg);
    
    toFundamentalRegion!(qaoa, sol.u)

    Γmin_interp, Emin_interp = sol.u, sol.objective

    return Γmin_interp, Emin_interp
end


function optimizeWithStrategy(qaoa::QAOA, 
    Γ0::Vector{T}, 
    pmax::Int,
    init::K,
    alg
    ) where {T<:Real, K<:InterpInitialization}

    p = length(Γ0) ÷ 2 
    @assert p < pmax

    energies_optima = zeros(T, pmax-p)
    params_optima   = Vector{Vector{T}}(undef, pmax-p)

    params_optima[1] = Γ0
    energies_optima[1] = qaoa(Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
    for t ∈ 2:(pmax-p)
        Γopt, Eopt = rollDown(qaoa, params_optima[t-1], init, alg)
        
        energies_optima[t] = Eopt
        params_optima[t] = Γopt

        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end
    return energies_optima, params_optima
end

# @doc raw"""
#     interpInitialization(Γp::Vector{Float64})

# Given an initial state `Γp::Vector{Float64}` of length `2p` it creates another vector
# `ΓInterp` of size ``2p+2`` with ``\gamma (\beta)`` components given by the following expression

# ```math
# \gamma^i_{p+1} = \frac{i-1}{p} \gamma^{i-1}_{p} + \frac{p-i+1}{p}\gamma^{i}_{p}
# ```
# and analogously for the ``\beta`` components.
# """
# function interpInitialization(Γ::Vector{T}) where T<:Real
#     p = length(Γ) ÷ 2
#     β = @view Γ[2:2:2p]
#     γ = @view Γ[1:2:2p]
    
#     βNew = map(x->((x-1)/p)*(x==1 ? 0 : β[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : β[x]), 1:p+1)
#     γNew = map(x->((x-1)/p)*(x==1 ? 0 : γ[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : γ[x]), 1:p+1)

#     ΓNew = zeros(T, 2*(p+1))
#     ΓNew[2:2:2(p+1)] = βNew
#     ΓNew[1:2:2(p+1)] = γNew

#     return ΓNew
# end

# @doc raw"""
#     rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    
# Starting from a local minima we construct a new vector using the INTERP initialization from which we perform the
# optimization. 

# # Arguments 
# * `qaoa::QAOA`: QAOA object 
# * `Γmin::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# # Optional
# * `method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))`: Default optimizer and linesearch choice. For more available choices see [*Optim.jl*](https://julianlsolvers.github.io/Optim.jl/stable/) 

# # Return
# * `result:Tuple`. The first element corresponds to the vector corresponding to which the algorithm converged to, and the second element is correponding energy_history
# """
# function rollDownInterp(qaoa::QAOA, Γmin::Vector{T}, 
#     alg) where {T<:Real}

#     ΓInterp = interpInitialization(Γmin)

#     sol = optimizeParameters(qaoa, ΓInterp, alg);
#     toFundamentalRegion!(qaoa, sol.u)
#     Γmin_interp, Emin_interp = sol.u, sol.objective
#     return Γmin_interp, Emin_interp
# end

# @doc raw"""
#     interpOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    
# Starting from a local minima `Γ0` at ``p=1`` it performs the `Interp` optimization strategy until the circuit depth `pmax` is reached.
# By default the `BFGS` optimizer is used. 

# # Arguments 
# * `qaoa::QAOA`: QAOA object 
# * `Γ0::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# # Optional
# * `method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))`: Default optimizer and linesearch choice. For more available choices see [*Optim.jl*](https://julianlsolvers.github.io/Optim.jl/stable/) 

# # Return
# * `result:Dict`. Dictionary with keys being `keys \in [1, pmax]` and values being a `Tuple{Float64, Vector{Float64}}` of cost function value and corresponding parameter.
# """
# function interpOptimize(qaoa::QAOA, 
#     Γ0::Vector{T}, 
#     pmax::Int,
#     alg
#     ) where {T<:Real}

#     listMinima = Dict{T, Tuple{T, Vector{T}}}()
#     p = length(Γ0) ÷ 2 
#     listMinima[p] = (qaoa(Γ0), Γ0)

#     iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
#     for t ∈ p+1:pmax
#         Γopt, Eopt = rollDownInterp(qaoa, listMinima[t-1][end], alg)
#         listMinima[t] = (Eopt, Γopt)
#         next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
#     end
#     return listMinima
# end