@doc raw"""
    interpInitialization(Γp::Vector{Float64})

Given an initial state `Γp::Vector{Float64}` of length `2p` it creates another vector
`ΓInterp` of size ``2p+2`` with ``\gamma (\beta)`` components given by the following expression

```math
\gamma^i_{p+1} = \frac{i-1}{p} \gamma^{i-1}_{p} + \frac{p-i+1}{p}\gamma^{i}_{p}
```
and analogously for the ``\beta`` components.
"""
function interpInitialization(Γ::Vector{T}) where T<:Real
    p = length(Γ) ÷ 2
    β = @view Γ[2:2:2p]
    γ = @view Γ[1:2:2p]
    
    βNew = map(x->((x-1)/p)*(x==1 ? 0 : β[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : β[x]), 1:p+1)
    γNew = map(x->((x-1)/p)*(x==1 ? 0 : γ[x-1]) + ((p-x+1)/p)*(x==p+1 ? 0 : γ[x]), 1:p+1)

    ΓNew = zeros(T, 2*(p+1))
    ΓNew[2:2:2(p+1)] = βNew
    ΓNew[1:2:2(p+1)] = γNew

    return ΓNew
end

@doc raw"""
    rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    
Starting from a local minima we construct a new vector using the INTERP initialization from which we perform the
optimization. 

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γmin::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# Optional
* `method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))`: Default optimizer and linesearch choice. For more available choices see [*Optim.jl*](https://julianlsolvers.github.io/Optim.jl/stable/) 

# Return
* `result:Tuple`. The first element corresponds to the vector corresponding to which the algorithm converged to, and the second element is correponding energy_history
"""
function rollDownInterp(qaoa::QAOA{P, H, M}, Γmin::Vector{T}; 
    setup=OptSetup()
    ) where {P, H, M, T<:Real}

    ΓInterp = interpInitialization(Γmin)

    Γmin_interp, Emin_interp = optimizeParameters(qaoa, ΓInterp, setup=setup);
    return Γmin_interp, Emin_interp
end

@doc raw"""
    interpOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))
    
Starting from a local minima `Γ0` at ``p=1`` it performs the `Interp` optimization strategy until the circuit depth `pmax` is reached.
By default the `BFGS` optimizer is used. 

# Arguments 
* `qaoa::QAOA`: QAOA object 
* `Γ0::Vector{Float64}`: Vector correponding to the local minimum from which we will construct the particular TS and then **roll down** from.

# Optional
* `method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))`: Default optimizer and linesearch choice. For more available choices see [*Optim.jl*](https://julianlsolvers.github.io/Optim.jl/stable/) 

# Return
* `result:Dict`. Dictionary with keys being `keys \in [1, pmax]` and values being a `Tuple{Float64, Vector{Float64}}` of cost function value and corresponding parameter.
"""
function interpOptimize(qaoa::QAOA{P, H, M}, 
    Γ0::Vector{T}, 
    pmax::Int; 
    setup=OptSetup()
    ) where {P, H, M, T<:Real}

    listMinima = Dict{T, Tuple{T, Vector{T}}}()
    p = length(Γ0) ÷ 2 
    listMinima[p] = (qaoa(Γ0), Γ0)

    iter = Progress(pmax-p; desc="Optimizing QAOA energy...")
    
    for t ∈ p+1:pmax
        Γopt, Eopt = rollDownInterp(qaoa, listMinima[t-1][end]; setup=setup)
        listMinima[t] = (Eopt, Γopt)
        next!(iter; showvalues = [(:Circuit_depth, t), (:Energy, Eopt)])
    end
    return listMinima
end