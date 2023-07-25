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