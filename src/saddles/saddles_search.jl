"""
    gradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}) where T <: Real

Calculate the squared norm of the gradient of the QAOA cost function at the given point.

# Arguments
- `qaoa::QAOA`: QAOA problem instance.
- `point::AbstractVector{T}`: Point in parameter space at which to compute the gradient squared norm.

# Returns
- `Float64`: Returns the squared norm of the gradient at the given point.
"""
function gradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}) where T <: Real
    grad = gradCostFunction(qaoa, point)
    return norm(grad)^2
end


"""
    optimizeGradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}; printout=false, 
        method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))) where T <: Real

Optimizes the squared norm of the gradient of the QAOA cost function given the initial parameter `point`.

# Arguments
- `qaoa::QAOA`: QAOA problem instance.
- `point::AbstractVector{T}`: Initial point in parameter space to start the optimization.
- `printout::Bool` (optional): If true, prints the final cost function value and gradient norm. Default is false.
- `method::Optim.AbstractOptimizer` (optional): Optimizer to use. Default is BFGS with backtracking linesearch of order 3.

# Returns
- `Tuple`: Returns a tuple containing the optimized parameters and the minimum cost.
"""
function optimizeGradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}; printout=false, 
    method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))
    ) where T <: Real

    gradCostFun(x::AbstractVector{T}) where T<:Real = gradSquaredNorm(qaoa, x)

    result = Optim.optimize(gradCostFun, point, autodiff = :forward, method=method)
    
    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)

    toFundamentalRegion!(qaoa, parameters)
    if printout
        gradientNorm = norm(gradCostFunction(qaoa, parameters))
        println("Optimization with final cost function value Loss=$(cost), and gradient norm |∇Loss|=$(gradientNorm)")
    end
    return parameters, cost
end


"""
    getStationaryPoints(
        qaoa::QAOA, 
        p::Integer,
        grid_of_points;
        printout = false, 
        threaded = false,
        method = Optim.BFGS(linesearch = Optim.BackTracking(order=3))
    )

Computes the stationary points of the QAOA cost function given an initial grid of points. The finer the grid the more points we should find (until saturation is reached)

The optimization is performed using the provided method (BFGS with backtracking linesearch by default). The function can operate either in single-threaded or multi-threaded mode.

# Arguments
- `qaoa::QAOA`: QAOA problem instance.
- `p::Integer`: Integer value related to the problem dimensionality.
- `grid_of_points`: Initial points in parameter space from which the optimization begins.
- `printout::Bool` (optional): If true, prints the final cost function value and gradient norm. Default is false.
- `threaded::Bool` (optional): If true, uses multi-threaded mode. Default is false.
- `method::Optim.AbstractOptimizer` (optional): Optimizer to use. Default is BFGS with backtracking linesearch of order 3.

# Returns
- `Tuple`: Returns a tuple containing the final energies and the corresponding parameters at the stationary points.
"""
function getStationaryPoints(qaoa::QAOA, p::Integer,
    grid_of_points;
    printout = false, threaded=false,
    method = Optim.BFGS(linesearch = Optim.BackTracking(order=3))
    )

    #grid_of_points = generate_grid(npoints, p)
    # temp_energy   = Float64(0)
    # temp_params   = zeros(Float64, 2*p)
    
    num_points = length(grid_of_points)
    converged_points   = zeros(2p, num_points)
    converged_energies = zeros(num_points)
    
    if threaded
        Threads.@threads for i in eachindex(grid_of_points)
            converged_points[:, i], _ = optimizeGradSquaredNorm(qaoa, collect(grid_of_points[i]), method=method, printout = printout)
            converged_energies[i] = qaoa(converged_points[:, i])
        end
    else
        for (i, point) in enumerate(grid_of_points)
            converged_points[:, i], _ = optimizeGradSquaredNorm(qaoa, collect(point), method=method, printout = printout)
            converged_energies[i] = qaoa(converged_points[:, i])
        end
    end

    return converged_energies, converged_points
end



"""
    gad(qaoa::QAOA, init_point::Vector{T}; niter = 500, η=0.01, tol=1e-5) where T<:Real

Perform the Gentlest Ascent Dynamics (GAD) optimization algorithm on a QAOA problem with the goal to find index-1 saddle points.

# Arguments
- `qaoa::QAOA`: a QAOA problem instance.
- `init_point::Vector{T}`: the initial point in parameter space, where `T` is a subtype of `Real`.
- `niter::Int=500`: the maximum number of iterations to perform (optional, default is 500).
- `η::Float64=0.01`: the step size for the GAD algorithm (optional, default is 0.01).
- `tol::Float64=1e-5`: the tolerance for the gradient norm. If the norm falls below this value, the algorithm stops (optional, default is 1e-5).

# Returns
- `point_history`: history of points during the iterations.
- `energ_history`: history of energy values during the iterations.
- `grad_history`: history of gradient norms during the iterations.

# Usage
```julia
point_history, energ_history, grad_history = gad(qaoa, init_point, niter=500, η=0.01, tol=1e-5)
"""
function gad(qaoa::QAOA, init_point::Vector{T}; niter = 500, η=0.01, tol=1e-5) where T<:Real
    p = length(init_point)
    
    point_temp = copy(init_point)
    
    point_history = Vector[]
    energ_history = Float64[]
    grad_history  = Float64[]
    
    v_hess = zeros(T, 2p)
    grad   = zeros(T, 2p)
    hess   = zeros(T, 2p, 2p)
    for i ∈ 1:niter
        # compute grad #
        grad = gradCostFunction(qaoa, point_temp)
        # compute hessian #
        hess = hessianCostFunction(qaoa, point_temp)
        v_hess = eigen(hess).vectors[:, 1]
        # perform one iteration
        point_temp .+= η*(-grad + 2*dot(grad, v_hess)*v_hess)
        toFundamentalRegion!(qaoa, point_temp)
        
        push!(point_history, point_temp)
        push!(energ_history, qaoa(point_temp), )
        push!(grad_history, gradCostFunction(qaoa, point_temp) |> norm)

        if grad_history[end] <= tol
            println("Algorithm converged at iteration iter=$(i)")
            break
        end
    end
    println("Maximum number of iterations reached niter=$(niter)")
    println("---- Gradient norm is ∇E = $(grad_history[end])")
    println("---- Energy of converged point E = $(energ_history[end])")
    return point_history, energ_history, grad_history
end


function modulatedNewton(
    qaoa::QAOA{P, H, M}, 
    Γ0::Vector{T},
    niter::Int=1_000
    ) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat}
    
    Γ = similar(Γ0)
    Γ .= Γ0 
    dim = length(Γ)

    gradTape = QAOALandscapes.GradientTape(qaoa)
    function g!(G,x)
        QAOALandscapes.gradient!(G, qaoa, gradTape, x)
    end
    f(x) = qaoa(x)
    ϵ = cbrt(eps(Float64))

    # allocate matrix h for storing the hessian, and vector g for storing the gradient
    h = zeros(T, dim, dim)
    g = zeros(T, dim)

    # to store the eigen-decomposition of the hessian
    vals = zeros(T, dim)
    vecs = zeros(T, dim, dim)
    
    @inbounds for i ∈ 1:niter
        g!(g, Γ)
        h .= hessianCostFunction(qaoa, Γ)

        vals, vecs = eigen(hermitianpart(h))
        vals .= abs.(vals)

        # to avoid NaNs of Infs!
        vals[vals .< ϵ] .+= ϵ

        # aiming for index-1 saddle

        h .= vecs * ((1 ./ vals) .* vecs')
        Γ .= Γ - h * g

        if norm(g) < 1e-5
            return Γ, qaoa(Γ), i
        end
    end 
    return Γ, qaoa(Γ), niter
end

function modulatedNewtonSaddles(qaoa::QAOA{P, H, M}, Γ0::Vector{T}, niter::Int=1_000) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat}
    # copy initial parameter
    Γ = similar(Γ0)
    Γ .= Γ0 
    # fold parameter to fundamental region
    toFundamentalRegion!(qaoa, Γ)
    dim = length(Γ)
    
    ϵ = cbrt(eps(Float64))

    # allocate matrix h for storing the hessian, and vector g for storing the gradient
    h = zeros(T, dim, dim)
    g = zeros(T, dim)

    # to store the eigen-decomposition of the hessian
    vals = zeros(T, dim)
    vecs = zeros(T, dim, dim)
    
    for _ ∈ 1:niter
        g .= gradCostFunction(qaoa, Γ)
        h .= hessianCostFunction(qaoa, Γ)
        
        vals, vecs = eigen(hermitianpart(h))
        vals .= abs.(vals)

        # to avoid NaNs of Infs!
        vals[vals .< ϵ] .+= ϵ

        # aiming for index-1 saddle
        vals[1] *= -1

        h .= vecs * Diagonal(1 ./ vals) * vecs'
        Γ .= Γ - h * g

        toFundamentalRegion!(qaoa, Γ)
        
        #push!(energies, qaoa(Γ))
        #push!(parameters, Γ)
        
        if norm(g) < 1e-5
            break
        end
    end 

    return Γ, qaoa(Γ)
end

function findMinimaAdam(qaoa::QAOA{P, H, M}, Γ0::Vector{T}, niter::Int=300) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat}
    gradTape = GradientTape(qaoa)
    function g!(G,x)
        gradient!(G, qaoa, gradTape, x)
    end
    result = Optim.optimize(qaoa, g!, Γ0, method=Adam(), iterations = niter)
    opt_param = Optim.minimizer(result)
    toFundamentalRegion!(qaoa, opt_param)
    return opt_param, Optim.minimum(result)
end

function optimizeIPNewton(qaoa::QAOA{P, H, M}, 
    Γ0::Vector{T}, 
    bounds::Tuple{V, V}
    ) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat, V<:Vector}
    
    gradTape = GradientTape(qaoa)
    function g!(G,x)
        gradient!(G, qaoa, gradTape, x)
    end
    function h!(H, x)
        H .= hessianCostFunction(qaoa, x)
    end

    dF = Optim.TwiceDifferentiable(qaoa, g!, h!, Γ0)
    low_params = bounds[1]
    upper_params = bounds[2]
    dfc = TwiceDifferentiableConstraints(low_params, upper_params)
    res = optimize(dF, dfc, Γ0, IPNewton())
    return Optim.minimizer(res), Optim.minimum(res) 
end

function optimizeFminbox(
    qaoa::QAOA{P, H, M}, 
    Γ0::Vector{T}, 
    bounds::Tuple{V, V}
    ) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat, V<:Vector}
    
    gradTape = GradientTape(qaoa)
    function g!(G,x)
        gradient!(G, qaoa, gradTape, x)
    end
    inner_optimizer = Optim.BFGS(linesearch=Optim.BackTracking(order=3))

    low_params = bounds[1]
    upper_params = bounds[2]

    res = optimize(qaoa, g!, low_params, upper_params, Γ0, Fminbox(inner_optimizer))
    return Optim.minimizer(res), Optim.minimum(res), Optim.iterations(res)
end

function warmOptimizeModulatedNewton(fun, vector_of_Γs::Matrix{T}) where {T}
    npoints = size(vector_of_Γs, 2)
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    number_of_iters = zeros(Int, npoints)
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x], number_of_iters[x]) = modulatedNewton(fun, vector_of_Γs[:, x])
        next!(prog)
    end
    return vector_of_energs, vector_of_params, number_of_iters
end

function warmOptimizeFminbox(fun, vector_of_Γs::Matrix{T}, bounds::Tuple{V, V}) where {T, V<:Vector}
    npoints = size(vector_of_Γs, 2)
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    number_of_iters = zeros(Int, npoints)
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x], number_of_iters[x]) = optimizeFminbox(fun, vector_of_Γs[:, x], bounds)
        next!(prog)
    end
    return vector_of_energs, vector_of_params, number_of_iters
end