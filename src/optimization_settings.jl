"""
    optimizeParameters(qaoa::QAOA, params::AbstractVector{T};
                       method = Optim.BFGS(linesearch = Optim.BackTracking(order=3)),
                       printout::Bool = false) where T<:Real

Optimizes the QAOA parameters using the specified optimization method and linesearch algorithm.

# Arguments
- `qaoa::QAOA`: A QAOA instance representing the problem to be solved.
- `params::AbstractVector{T}`: The initial guess for the optimization parameters, where T is a subtype of Real.

# Keyword Arguments
- `method`: The optimization method to be used (default: `Optim.BFGS(linesearch = Optim.BackTracking(order=3))`).
- `printout::Bool`: Whether to print optimization progress information (default: `false`).

# Returns
- A tuple containing the optimized parameters and the corresponding minimum cost function value.

# Examples
```julia
result = optimizeParameters(qaoa, params, method=Optim.BFGS(linesearch=Optim.HagerZhang()), printout=true)
```
"""
function optimizeParameters(
    qaoa::QAOA, params::AbstractVector{T};
    method = Optim.BFGS(linesearch = Optim.BackTracking(order=3)),
    printout=false, diffMode=:adjoint) where T<:Real

    if diffMode == :adjoint
        function g!(G,x)
            G .= gradCostFunction(qaoa, x)
        end
        result = Optim.optimize(qaoa, g!, params, method=method)
    elseif diffMode == :forward
        result = Optim.optimize(qaoa, params, method = method, autodiff = diffMode)
    else
        throw(ArgumentError("Wrong diffMode = $(diffMode) given. Supported modes are :adjoint (default) and :forward"))
    end
    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)

    toFundamentalRegion!(qaoa, parameters)
    if printout
        gradientNorm = norm(gradCostFunction(qaoa, parameters))
        print("Optimization with final cost function value E=$(cost), and gradient norm |∇E|=$(gradientNorm)")
    end
    return parameters, cost
end

@doc raw"""
    optimizeParameters(::Val{:Fourier}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `BFGS`
optimizer. Here we use the alternative "Fourier" initialization, where instead of optimizing the usual (γ, β) parameters
we optimize their frecuency components ``(u_{\gamma}, u_{\beta})``.

# Arguments

* `Val(:BFGS)`: For using the BFGS. Alternatively, `Val(:GD)` for using the `ADAM` optimizer
* `qaoa:QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `parameters::Vector{Float64}`: Optimal parameter obtained
* `cost::Float64`: Value of the cost function for the optimal parameter obtained.
"""
function optimizeParameters(::Val{:Fourier}, qaoa::QAOA, params::AbstractVector{T};
    method = Optim.BFGS(linesearch = Optim.BackTracking(order=3)), 
    printout=false) where T<:Real
    
    f(x::Vector{Float64})  = qaoa(fromFourierParams(x))
    function ∇f!(G, x::Vector{Float64}) 
        G .= gradCostFunctionFourier(qaoa, x)
    end
    result = Optim.optimize(f, ∇f!, params, method=method)
    
    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)

    if printout
        gradientNorm = gradCostFunctionFourier(qaoa, parameters) |> norm
        print("Optimization with final cost function value E=$(cost), and gradient norm |∇E|=$(gradientNorm)")
    end

    return fromFourierParams(parameters), cost
end

@doc raw"""
    getInitParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)

Given a `QAOA` object it performs a grid search on a region of the two dimensional space spanned by ``\{ \gamma_1, \beta_1\}``
The ``\beta_1`` component is in the interval ``[-\pi/4, \pi/4]`` while the ``\gamma_1`` part is in the ``(0, \pi/4]`` for 3RRG
or ``(0, \pi/2]`` for dRRG (with ``d\neq 3``). 

We then launch the `QAOA` optimization procedure from the point in the 2-dimensional grid with the smallest cost function value.

# Returns
* 3-Tuple containing: 1.) the cost function grid, 2.) the optimal parameter, and 3.) the optimal energy
"""
function getInitialParameter(qaoa::QAOA; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)), spacing = 0.01, gradTol = 1e-6)
    βIndex  = collect(-π/4:spacing:π/4)
    
    isWeightedG  = typeof(qaoa.graph) <: SimpleWeightedGraph
    if  QAOALandscapes.isdRegularGraph(qaoa.graph, 3) && !isWeightedG
        γIndex  = collect(0.0:(spacing/2):π/4)
    else
        spacing *= 2
        γIndex  = collect(-π/2:spacing:π/2)
    end
    
    energy  = zeros(length(γIndex), length(βIndex))
    for j in eachindex(βIndex)
        for i in eachindex(γIndex)
            energy[i,j] = qaoa([γIndex[i], βIndex[j]])
        end
    end
    pos = argmin(energy)
    Γ   = [γIndex[pos[1]], βIndex[pos[1]]] 

    gradNormGridMin = norm(gradCostFunction(qaoa, Γ))
    if gradNormGridMin > gradTol
        newParams, newEnergy = optimizeParameters(qaoa, Γ, method=method)
        println("Convergence reached. Energy = $(newEnergy), |∇E| = $(norm(gradCostFunction(qaoa, newParams)))")
    end
    return energy, newParams, newEnergy
end