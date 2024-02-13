struct OptSetup{T<:Optim.AbstractOptimizer}
    method::T
    options::Union{Optim.Options, Nothing}
    printout::Bool
    diff_mode::Symbol
end

function OptSetup(method::T; printout=true, diff_mode=:adjoint) where T<:Optim.AbstractOptimizer
    return OptSetup(method, nothing, printout, diff_mode)
end

function OptSetup(method::T, options::S; printout=true, diff_mode=:adjoint) where {T<:Optim.AbstractOptimizer, S<:Optim.Options}
    return OptSetup(method, options, printout, diff_mode)
end

function OptSetup(; printout=true, diff_mode=:adjoint)
    return OptSetup(
        Optim.BFGS(linesearch=Optim.BackTracking(order=3)),
        nothing,
        printout,
        diff_mode
    )
end

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

!!! Important
- `diffMode` = :adjoint is the default but is only meant to be used for 1st and quasi 2nd order method. Use `diffMode=:forward` if you want to use something like Newton's methdd 

# Returns
- A tuple containing the optimized parameters and the corresponding minimum cost function value.

# Examples
```julia
result = optimizeParameters(qaoa, params, method=Optim.BFGS(linesearch=Optim.HagerZhang()), printout=true)
```
"""
function optimizeParametersTaped(
    qaoa::QAOA{T1, T}, 
    params::AbstractVector{T};
    setup = OptSetup()
    ) where {T1<:AbstractGraph, T<:Real}
    
    type_optim = typeof(setup.method)
    @assert in(setup.diff_mode, [:forward, :adjoint, :notdiff])
    
    if type_optim <: Optim.SecondOrderOptimizer
        if isa(setup.options, Nothing)
            result = Optim.optimize(qaoa, params, setup.method, autodiff = :forward)
        else
            result = Optim.optimize(qaoa, params, setup.method, setup.options, autodiff = :forward) 
        end
    
    elseif type_optim <: Optim.FirstOrderOptimizer
        function g!(G,x)
            G .= gradCostFunction(qaoa, x)
        end
        if isa(setup.options, Nothing)
            if setup.diff_mode == :forward
                result = Optim.optimize(qaoa, params, setup.method, autodiff = :forward)
            elseif setup.diff_mode == :adjoint
                result = Optim.optimize(qaoa, g!, params, setup.method)
            else
                throw(ArgumentError("Need to define an autodiff method for $(setup.method)"))
            end
        else
            result = Optim.optimize(qaoa, g!, params, setup.method, setup.options)
        end
    else
        result = Optim.optimize(qaoa, params, setup.method, setup.options)
    end

    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)
    convergence_info = Optim.converged(result)

    toFundamentalRegion!(qaoa, parameters)
    if !convergence_info
        if qaoa(params) < cost
            throw(AssertionError("Optimization did not converged. Final gradient norm is |∇E|=$(norm(gradCostFunction(qaoa, parameters)))"))
        elseif setup.printout
            println("Optimization did not converged but energy decreased")
        end
    end
    return parameters, cost
end

function optimizeParameters(
    qaoa::QAOA{T1, T, T3}, 
    params::Vector{T};
    setup = OptSetup()
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    type_optim = typeof(setup.method)
    if type_optim <: Optim.FirstOrderOptimizer
        gradTape = GradientTape(qaoa)
        function g!(G,x)
            gradient!(G, qaoa, gradTape, x)
        end
        if isa(setup.options, Nothing)
            if setup.diff_mode == :forward
                result = Optim.optimize(qaoa, params, setup.method, autodiff = :forward)
            elseif setup.diff_mode == :adjoint
                result = Optim.optimize(qaoa, g!, params, setup.method)
            else
                throw(ArgumentError("Need to define an autodiff method for $(setup.method)"))
            end
        else
            result = Optim.optimize(qaoa, g!, params, setup.method, setup.options)
        end
    else 
        throw(ArgumentError("Only supported with 1st order optimizers"))
    end

    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)
    convergence_info = Optim.converged(result)

    toFundamentalRegion!(qaoa, parameters)
    if !convergence_info
        if qaoa(params) < cost
            throw(AssertionError("Optimization did not converged. Final gradient norm is |∇E|=$(norm(gradTape.value))"))
        elseif setup.printout
            println("Optimization did not converged but energy decreased")
        end
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
function optimizeParameters(::Val{:Fourier}, 
    qaoa::QAOA{T1, T, T3}, 
    params::AbstractVector{T};
    setup = OptSetup()
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    f(x::Vector{S}) where S<:Real = qaoa(fromFourierParams(x))
    function ∇f!(G, x::Vector{S}) where S<:Real
        G .= gradCostFunctionFourier(qaoa, x)
    end

    type_optim = typeof(setup.method)
    @assert in(setup.diff_mode, [:adjoint, :notdiff])
    @assert !(type_optim <: Optim.SecondOrderOptimizer)
    
    if type_optim <: Optim.FirstOrderOptimizer
        if isa(setup.options, Nothing)
            result = Optim.optimize(f, ∇f!, params, setup.method)
        else
            result = Optim.optimize(f, ∇f!, params, setup.method, setup.options)
        end
    else
        result = Optim.optimize(f, params, setup.method, setup.options)
    end

    parameters = fromFourierParams(Optim.minimizer(result))
    cost       = Optim.minimum(result)
    convergence_info = Optim.converged(result)
    toFundamentalRegion!(qaoa, parameters)

    if !convergence_info
        if f(params) < cost
            throw(AssertionError("Optimization did not converged. Final gradient norm is |∇E|=$(norm(gradCostFunction(qaoa, parameters)))"))
        elseif setup.printout
            println("Optimization did not converged but energy decreased")
        end
    end

    return parameters, cost
end

"""
    optimizeParametersSlice(qaoa::QAOA, ΓTs::Vector{T}, u::Vector{T};
                            method=Optim.BFGS(linesearch=LineSearches.BackTracking(order=3))) where T<:Real

Optimize the parameters of the QAOA along the index-1 direction of the transition state `ΓTs`. 

# Arguments

- `qaoa::QAOA`: QAOA instance 
- `ΓTs::Vector{T}`: Initial parameter vector. We assume that `ΓTs` is an index-1 saddle point and the `u` is the index-1 direction
- `u::Vector{T}`: Direction along which the `QAOA` cost function is going to be optimized. 
- `method`: The optimization method to be used for the parameter optimization.
    Default: `Optim.BFGS(linesearch=LineSearches.BackTracking(order=3))`.

# Returns

- `f_vals::Vector{T}`: The minimum objective function values obtained from the optimization process for the negative and positive parameter shift cases.
- `x_vals::Vector{Vector{T}}`: The corresponding parameter values that yield the minimum objective function values.

"""
function optimizeParametersSlice(qaoa::QAOA{T1,T, T3}, 
    Γmin::Vector{T}, 
    ig::Integer,
    gsIndex::Vector{Int};
    tsType = "symmetric"
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    ΓTs = transitionState(Γmin, ig, tsType=tsType)
    u   = getNegativeHessianEigvec(qaoa, Γmin, ig, tsType=tsType)["eigvec_approx"] |> Array

    energ(x) = qaoa(ΓTs + u*x[1])
    distn(x) = -gsFidelity(qaoa, ΓTs + u*x[1], gsIndex)[1]
    # Set limits of search #
    lower = T.([(-1)])
    upper = T.([1])
    
    # Set initial parameters
    x0_p = T.([0.01])
    x0_m = T.([-0.01])
    
    # Set inner optimizer #
    inner_optimizer = Optim.BFGS(linesearch=LineSearches.BackTracking(order=3))
    
    energ_res_m = optimize(energ, lower, upper, x0_m, Fminbox(inner_optimizer))
    energ_res_p = optimize(energ, lower, upper, x0_p, Fminbox(inner_optimizer))

    distn_res_m = optimize(distn, lower, upper, x0_m, Fminbox(inner_optimizer))
    distn_res_p = optimize(distn, lower, upper, x0_p, Fminbox(inner_optimizer))
    
    result = Dict(
        "energy" => (val = [Optim.minimum(energ_res_m), Optim.minimum(energ_res_p)], x_opt = vcat([Optim.minimizer(energ_res_m), Optim.minimizer(energ_res_p)]...)),
        "fidelity" => (val = [-Optim.minimum(distn_res_m), -Optim.minimum(distn_res_p)], x_opt = vcat([Optim.minimizer(distn_res_m), Optim.minimizer(distn_res_p)]...))
    )
    
    return result
end


@doc raw"""
    getInitialParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)

Given a `QAOA` object it performs a grid search on a region of the two dimensional space spanned by ``\{ \gamma_1, \beta_1\}``
The ``\beta_1`` component is in the interval ``[-\pi/4, \pi/4]`` while the ``\gamma_1`` part is in the ``(0, \pi/4]`` for 3RRG
or ``(0, \pi/2]`` for dRRG (with ``d\neq 3``). 

We then launch the `QAOA` optimization procedure from the point in the 2-dimensional grid with the smallest cost function value.

# Returns
* 3-Tuple containing: 1.) the cost function grid, 2.) the optimal parameter, and 3.) the optimal energy
"""
function getInitialParameter(qaoa::QAOA{T1, T, T3}; 
    setup=OptSetup(), 
    num_points=20, 
    ) where {T1<:AbstractGraph, T<:Real, T3<:AbstractBackend}
    
    initial_points = rand(T, 2, num_points)
    energies_points = zeros(T, num_points)
    params_points  = zeros(T, 2, num_points)

    for i ∈ 1:num_points
        (params_points[:, i], energies_points[i]) = optimizeParameters(qaoa, Vector(initial_points[:, i]), setup=setup)
    end
    
    (Einit, index) = findmin(energies_points)
    Γ = Vector(params_points[:, index])
    toFundamentalRegion!(qaoa, Γ)

    return Γ, Einit
end