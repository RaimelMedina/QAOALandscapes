var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [QAOALandscapes]\nOrder   = [:type, :function]","category":"page"},{"location":"api/#QAOALandscapes.QAOA","page":"API","title":"QAOALandscapes.QAOA","text":"QAOA{K<:AbstractProblem, T<:AbstractVector, M <: AbstractMixer}\n    N::Int\n    problem::K\n    HC::T\n    mixer::M\n    initial_state::T\nend\n\nDefinition of the QAOA object.\n\n\n\n\n\n","category":"type"},{"location":"api/#QAOALandscapes.QAOA-Union{Tuple{ClassicalProblem{R}}, Tuple{R}} where R<:Real","page":"API","title":"QAOALandscapes.QAOA","text":"QAOA(cp::ClassicalProblem{R}) where R<:Real\nQAOA(cp::ClassicalProblem{R}, ham::Vector{Complex{R}}, mixer::AbstractMixer) where R<:Real\nQAOA(cp::ClassicalProblem{R}, ham::AbstractGPUArray{Complex{R}}, mixer::AbstractMixer) where R<:Real\n\nAvailable constructors for the QAOA object \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.computationalBasisWeights-Tuple{Any, Any}","page":"API","title":"QAOALandscapes.computationalBasisWeights","text":"computationalBasisWeights(ψ, equivClasses)\n\nComputes the computational basis weights for a given state vector ψ according to the provided equivalence classes, by summing up the squared magnitudes of the elements with the same equivalence class. Here ψ and equivClasses are ment to live in the same Hilbert space basis. \n\nArguments\n\nψ: The state vector.\nequivClasses: A Vector of Vectors, where each inner Vector contains the indices of the elements belonging to the same equivalence class.\n\nReturns\n\nbasis_weights::Vector{Float64}: A Vector containing the summed squared magnitudes of ψ for each equivalence class.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.energyVariance-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, AbstractVector{T}}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.energyVariance","text":"energyVariance(q::QAOA{P, H, M}, Γ::AbstractVector{T}) where {P, H, M, T<:Real}\nenergyVariance(q::QAOA{P, H, M}, ψ::AbstractVector{Complex{T}}) where {P, H, M, T}\n\nComputes the energy variance of the cost Hamiltonian H_C in the QAOA state.  Alternatively, computes the energy variance of the cost Hamiltonian in a given state psi:\n\n    mathrmvar_GammaH_C = langle Gamma^p H_C^2Gamma^p rangle-langle Gamma^p H_CGamma^p rangle^2\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.fourierOptimize-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Int64}, Tuple{QAOA{P, H, M}, Vector{T}, Int64, Int64}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.fourierOptimize","text":"fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int)\n\nStarting from a local minima Γ0 at p=1 it performs the Fourier optimization strategy until the circuit depth pmax is reached. By default the BFGS optimizer is used. \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓ0::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Dict. Dictionary with keys being keys \\in [1, pmax] and values being a Tuple{Float64, Vector{Float64}} of cost function value and corresponding parameter.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.gad-Union{Tuple{T}, Tuple{QAOA, Vector{T}}} where T<:Real","page":"API","title":"QAOALandscapes.gad","text":"gad(qaoa::QAOA, init_point::Vector{T}; niter = 500, η=0.01, tol=1e-5) where T<:Real\n\nPerform the Gentlest Ascent Dynamics (GAD) optimization algorithm on a QAOA problem with the goal to find index-1 saddle points.\n\nArguments\n\nqaoa::QAOA: a QAOA problem instance.\ninit_point::Vector{T}: the initial point in parameter space, where T is a subtype of Real.\nniter::Int=500: the maximum number of iterations to perform (optional, default is 500).\nη::Float64=0.01: the step size for the GAD algorithm (optional, default is 0.01).\ntol::Float64=1e-5: the tolerance for the gradient norm. If the norm falls below this value, the algorithm stops (optional, default is 1e-5).\n\nReturns\n\npoint_history: history of points during the iterations.\nenerg_history: history of energy values during the iterations.\ngrad_history: history of gradient norms during the iterations.\n\nUsage\n\n```julia pointhistory, energhistory, gradhistory = gad(qaoa, initpoint, niter=500, η=0.01, tol=1e-5)\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.geometricTensor-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, H}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T<:Real}","page":"API","title":"QAOALandscapes.geometricTensor","text":"geometricTensor(qaoa::QAOA, params::Vector{T}, ψ0::AbstractVector{Complex{T}}) where T<: Real\n\nCompute the geometricTensor of the QAOA cost function using adjoint (a reverse-mode) differentiation. We implement the algorithm  proposed in this reference\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getEquivalentClasses-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Real","page":"API","title":"QAOALandscapes.getEquivalentClasses","text":"getEquivalentClasses(vec::Vector{T}; sigdigits = 5) where T <: Real\n\nComputes the equivalence classes of the elements of the input vector vec The elements are rounded to a certain number of significant digits (default is 5) to group the states with approximately equal values.\n\nArguments\n\nvec::Vector{T<:Real}\nsigdigits=5: Significant digits to which energies are rounded.\n\nReturns\n\ndata_states::Dict{Float64, Vector{Int}}: Keys are unique elements (rounded) and values corresponds to the index of elements with the same key.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getHessianIndex-Union{Tuple{T}, Tuple{QAOA, Vector{T}}} where T<:Real","page":"API","title":"QAOALandscapes.getHessianIndex","text":"getHessianIndex(qaoa::QAOA, Γ::AbstractVector{T}; checks=true, tol=1e-6) where T<:Real\n\nCalculate the Hessian index of a stationary (it checks the gradient norm) point of the QAOA energy function\n\nArguments\n\nqaoa: a QAOA object.\nΓ: a vector of parameters.\n\nKeyword Arguments\n\nchecks=true: a boolean to decide whether to check if the gradient of the cost function is smaller than a certain tolerance.\ntol=1e-6: a tolerance level for the gradient of the cost function.\n\nOutput\n\nReturns the Hessian index, i.e., the number of negative eigenvalues of the Hessian matrix.\n\nNotes\n\nThe function first calculates the gradient of the cost function for the given qaoa and Γ. If checks=true, it asserts that the norm of this gradient is less than tol. It then calculates the Hessian matrix and its eigenvalues, and returns the count of eigenvalues less than zero.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getInitialParameter-Union{Tuple{QAOA{P, H, M}}, Tuple{M}, Tuple{H}, Tuple{P}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer}","page":"API","title":"QAOALandscapes.getInitialParameter","text":"getInitialParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)\n\nGiven a QAOA object it performs a grid search on a region of the two dimensional space spanned by  gamma_1 beta_1 The beta_1 component is in the interval -pi4 pi4 while the gamma_1 part is in the (0 pi4 for 3RRG or (0 pi2 for dRRG (with dneq 3). \n\nWe then launch the QAOA optimization procedure from the point in the 2-dimensional grid with the smallest cost function value.\n\nReturns\n\n3-Tuple containing: 1.) the cost function grid, 2.) the optimal parameter, and 3.) the optimal energy\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getNegativeHessianEigval-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Int64}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.getNegativeHessianEigval","text":"getNegativeHessianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType=\"symmetric\")\n\nComputes the approximation to the minimum (negative) eigenvalue of the Hessian at the TS obtained by padding with zeros the local minimum Γmin. The transition state is completely specified by the index of the γ component ig, and the  type of transition states (\"symmetric\" or \"non_symmetric\"). The cost of obtaining this approximate eigenvalue is basically the cost of computing two matrix elements of a Hessian.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getNegativeHessianEigvec-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Int64}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.getNegativeHessianEigvec","text":"getNegativeHessianEigvec(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType=\"symmetric\", doChecks=false)\n\nComputes the approximate eigenvalue and index-1 eigenvector of the Hessian at the transition state obtained from the local minimum  Γmin. It is completely specified by the parameters iγ and tsType=\"symmetric\". If the optional parameter doChecks=false is set to true, then the function also returns the relative error in estimating the true eigenvalue as well as the inner product between the approximate and true eigenvector\n\nArguments\n\nqaoa::QAOA: QAOA object\nΓmin::Vector{Float64}: The vector corresponding to a local minimum of QAOAₚ. \nig::Int: Index of the γ component where the zeros are inserted. \ntsType=\"symmetric\": In this case, the index of the β component is equal to ig. Otherwise, the β index is ig-1.\n\nOptional arguments\n\ndoChecks=false: In this case the function returns a dictionary with keys eigvec_approx and eigval_approx. If set to true it has additional keys => change_basis, eigvec_fidelity and eigval_error\n\nReturns\n\nresult::Dict Dictionary with the following keys: eigvec_approx, eigval_approx. If doChecks=true the following additional keys are available: change_basis, eigvec_fidelity and eigval_error\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getQAOAState-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, AbstractVector{T}, H}} where {P, H, M, T}","page":"API","title":"QAOALandscapes.getQAOAState","text":"getQAOAState(q::QAOA{P, H, M}, Γ::AbstractVector{T}, ψ0::H) where {P, H, M, T}\n\nConstruct the QAOA state. The main difference here is that it uses the state ψ0 as initial state instead of +rangle. That is:\n\n    Gamma^p rangle = U(Gamma^p) psi_0rangle\n\nwith\n\n    U(Gamma^p) = prod_l=1^p e^-i H_B beta_2l e^-i H_C gamma_2l-1\n\nand H_B H_C corresponding to the mixing and cost Hamiltonian respectively.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getStateProjection-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Vector{Int64}}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.getStateProjection","text":"getStateProjection(qaoa::QAOA, params, stateIndex::Vector{Int64})\n\nCalculates the projection of the QAOA state onto the state subspace determined by gsIndex. It also returns the corresponding orthogonal complement.  The QAOA state is determined by the given parameters params.\n\nArguments\n\nqaoa::QAOA: QAOA object.\nparams: Parameters determining the QAOA state.\nstateIndex::Vector{Int64}: Indices of the ground state components.\n\nReturns\n\nnormState: Norm of the projection of the QAOA state onto the state subspace given by stateIndex.\nnormState_perp: Norm of the projection of the QAOA state onto the orthogonal complement of the specificied state subspace.\nψIndex: Normalized projection of the QAOA state onto the state subspace.\nψIndex_perp: Normalized projection of the QAOA state onto the orthogonal complement of the state subspace.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getStationaryPoints-Tuple{QAOA, Integer, Any}","page":"API","title":"QAOALandscapes.getStationaryPoints","text":"getStationaryPoints(\n    qaoa::QAOA, \n    p::Integer,\n    grid_of_points;\n    printout = false, \n    threaded = false,\n    method = Optim.BFGS(linesearch = Optim.BackTracking(order=3))\n)\n\nComputes the stationary points of the QAOA cost function given an initial grid of points. The finer the grid the more points we should find (until saturation is reached)\n\nThe optimization is performed using the provided method (BFGS with backtracking linesearch by default). The function can operate either in single-threaded or multi-threaded mode.\n\nArguments\n\nqaoa::QAOA: QAOA problem instance.\np::Integer: Integer value related to the problem dimensionality.\ngrid_of_points: Initial points in parameter space from which the optimization begins.\nprintout::Bool (optional): If true, prints the final cost function value and gradient norm. Default is false.\nthreaded::Bool (optional): If true, uses multi-threaded mode. Default is false.\nmethod::Optim.AbstractOptimizer (optional): Optimizer to use. Default is BFGS with backtracking linesearch of order 3.\n\nReturns\n\nTuple: Returns a tuple containing the final energies and the corresponding parameters at the stationary points.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.gradCostFunction-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, AbstractVector{T}}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T}","page":"API","title":"QAOALandscapes.gradCostFunction","text":"gradCostFunction(qaoa::QAOA, params::Vector{T}) where T<: Real\n\nCompute the gradient of the QAOA cost function using adjoint (a reverse-mode) differentiation. We implement the algorithm  proposed in this reference. https://arxiv.org/pdf/2011.02991.pdf\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.gradSquaredNorm-Union{Tuple{T}, Tuple{QAOA, AbstractVector{T}}} where T<:Real","page":"API","title":"QAOALandscapes.gradSquaredNorm","text":"gradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}) where T <: Real\n\nCalculate the squared norm of the gradient of the QAOA cost function at the given point.\n\nArguments\n\nqaoa::QAOA: QAOA problem instance.\npoint::AbstractVector{T}: Point in parameter space at which to compute the gradient squared norm.\n\nReturns\n\nFloat64: Returns the squared norm of the gradient at the given point.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.hessianCostFunction-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T<:Real}","page":"API","title":"QAOALandscapes.hessianCostFunction","text":"hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real\n\nComputes the cost function Hessian at the point Gamma in parameter space.  The computation is done analytically since it has proven to be faster than the previous implementation using ForwardDiff.jl package\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.interpInitialization-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Real","page":"API","title":"QAOALandscapes.interpInitialization","text":"interpInitialization(Γp::Vector{Float64})\n\nGiven an initial state Γp::Vector{Float64} of length 2p it creates another vector ΓInterp of size 2p+2 with gamma (beta) components given by the following expression\n\ngamma^i_p+1 = fraci-1p gamma^i-1_p + fracp-i+1pgamma^i_p\n\nand analogously for the beta components.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.interpOptimize-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Int64}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.interpOptimize","text":"interpOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))\n\nStarting from a local minima Γ0 at p=1 it performs the Interp optimization strategy until the circuit depth pmax is reached. By default the BFGS optimizer is used. \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓ0::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nOptional\n\nmethod=Optim.BFGS(linesearch = Optim.BackTracking(order=3)): Default optimizer and linesearch choice. For more available choices see Optim.jl \n\nReturn\n\nresult:Dict. Dictionary with keys being keys \\in [1, pmax] and values being a Tuple{Float64, Vector{Float64}} of cost function value and corresponding parameter.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeGradSquaredNorm-Union{Tuple{T}, Tuple{QAOA, AbstractVector{T}}} where T<:Real","page":"API","title":"QAOALandscapes.optimizeGradSquaredNorm","text":"optimizeGradSquaredNorm(qaoa::QAOA, point::AbstractVector{T}; printout=false, \n    method=Optim.BFGS(linesearch = Optim.BackTracking(order=3))) where T <: Real\n\nOptimizes the squared norm of the gradient of the QAOA cost function given the initial parameter point.\n\nArguments\n\nqaoa::QAOA: QAOA problem instance.\npoint::AbstractVector{T}: Initial point in parameter space to start the optimization.\nprintout::Bool (optional): If true, prints the final cost function value and gradient norm. Default is false.\nmethod::Optim.AbstractOptimizer (optional): Optimizer to use. Default is BFGS with backtracking linesearch of order 3.\n\nReturns\n\nTuple: Returns a tuple containing the optimized parameters and the minimum cost.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParameters-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T<:Real}","page":"API","title":"QAOALandscapes.optimizeParameters","text":"optimizeParameters(qaoa::QAOA, params::AbstractVector{T};\n                   method = Optim.BFGS(linesearch = Optim.BackTracking(order=3)),\n                   printout::Bool = false) where T<:Real\n\nOptimizes the QAOA parameters using the specified optimization method and linesearch algorithm.\n\nArguments\n\nqaoa::QAOA: A QAOA instance representing the problem to be solved.\nparams::AbstractVector{T}: The initial guess for the optimization parameters, where T is a subtype of Real.\n\nKeyword Arguments\n\nmethod: The optimization method to be used (default: Optim.BFGS(linesearch = Optim.BackTracking(order=3))).\nprintout::Bool: Whether to print optimization progress information (default: false).\n\n!!! Important\n\ndiffMode = :adjoint is the default but is only meant to be used for 1st and quasi 2nd order method. Use diffMode=:forward if you want to use something like Newton's methdd \n\nReturns\n\nA tuple containing the optimized parameters and the corresponding minimum cost function value.\n\nExamples\n\nresult = optimizeParameters(qaoa, params, method=Optim.BFGS(linesearch=Optim.HagerZhang()), printout=true)\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParameters-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{Val{:Fourier}, QAOA{P, H, M}, AbstractVector{T}}, Tuple{Val{:Fourier}, QAOA{P, H, M}, AbstractVector{T}, Int64}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T<:Real}","page":"API","title":"QAOALandscapes.optimizeParameters","text":"optimizeParameters(::Val{:Fourier}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)\n\nPerform optimization of the QAOA using the gradient descent algorithm with the BFGS optimizer. Here we use the alternative \"Fourier\" initialization, where instead of optimizing the usual (γ, β) parameters we optimize their frecuency components (u_gamma u_beta).\n\nArguments\n\nVal(:BFGS): For using the BFGS. Alternatively, Val(:GD) for using the ADAM optimizer\nqaoa:QAOA: QAOA object\nΓ0::Vector{Float64}: Initial point from where the optimization starts\n\nKeyword arguments\n\nprintout=false: Whether if we print something during the optimization or not\n\nReturns\n\nIt returns a tuple containing the following information\n\nparameters::Vector{Float64}: Optimal parameter obtained\ncost::Float64: Value of the cost function for the optimal parameter obtained.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParametersSlice-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Integer, Vector{Int64}}} where {P<:AbstractProblem, H<:(AbstractVector), M<:AbstractMixer, T<:Real}","page":"API","title":"QAOALandscapes.optimizeParametersSlice","text":"optimizeParametersSlice(qaoa::QAOA, ΓTs::Vector{T}, u::Vector{T};\n                        method=Optim.BFGS(linesearch=LineSearches.BackTracking(order=3))) where T<:Real\n\nOptimize the parameters of the QAOA along the index-1 direction of the transition state ΓTs. \n\nArguments\n\nqaoa::QAOA: QAOA instance \nΓTs::Vector{T}: Initial parameter vector. We assume that ΓTs is an index-1 saddle point and the u is the index-1 direction\nu::Vector{T}: Direction along which the QAOA cost function is going to be optimized. \nmethod: The optimization method to be used for the parameter optimization.   Default: Optim.BFGS(linesearch=LineSearches.BackTracking(order=3)).\n\nReturns\n\nf_vals::Vector{T}: The minimum objective function values obtained from the optimization process for the negative and positive parameter shift cases.\nx_vals::Vector{Vector{T}}: The corresponding parameter values that yield the minimum objective function values.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.permuteHessian-Tuple{Type{<:Real}, Int64, Int64}","page":"API","title":"QAOALandscapes.permuteHessian","text":"permuteHessian(depth::Int, i::Int; tsType=\"symmetric\")\n\nComputes the permutation that takes the Hessian at a particular transition state into the form described in the paper Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.\n\nArguments\n\ndepth::Int: Circuit depth of the transition state.\ni::Int: Index of the γ component at which the zero is added. If tsType='symmetric' then β=i, otherwise if tsType='non_symmetric' β=i-1.\n\nReturn\n\nlistOfIndices::Vector{Int64}: List of indices correponding to the arrangement of the new basis elements.\npermMat::Matrix{Float64}: Matrix implementing the desired permutation.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.permuteHessian-Union{Tuple{T}, Tuple{Matrix{T}, Int64}} where T<:Real","page":"API","title":"QAOALandscapes.permuteHessian","text":"permuteHessian(H::AbstractArray{Float64,2}, i::Int; tsType=\"symmetric\")\n\nComputes the permutation that takes the Hessian at a particular transition state into the form described in the paper Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.\n\nArguments\n\nH::AbstractArray{Float64,2}: Hessian at the transition state in the original basis.\ni::Int: Index of the γ component at which the zero is added. If tsType='symmetric' then β=i, otherwise if tsType='non_symmetric' β=i-1.\n\nReturn\n\nlistOfIndices::Vector{Int64}: List of indices correponding to the arrangement of the new basis elements.\npermMat::Matrix{Float64}: Matrix implementing the desired permutation.\nHTransformed::Matrix{Float64}: Transformed Hessian at the transition state. Specifically, we have that H_mathoprmperm=PHP^-1.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownInterp-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.rollDownInterp","text":"rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; method=Optim.BFGS(linesearch = Optim.BackTracking(order=3)))\n\nStarting from a local minima we construct a new vector using the INTERP initialization from which we perform the optimization. \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nOptional\n\nmethod=Optim.BFGS(linesearch = Optim.BackTracking(order=3)): Default optimizer and linesearch choice. For more available choices see Optim.jl \n\nReturn\n\nresult:Tuple. The first element corresponds to the vector corresponding to which the algorithm converged to, and the second element is correponding energy_history\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownTS-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.rollDownTS","text":"rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, optim=Val(:BFGS))\n\nStarting from a local minima we construct all transition states (a total of 2p+1 of them). From each of the transition states, we construct two new vectors \n\nGamma^0_p = Gamma_rmTS + epsilon hate_rmmin \n\nGamma^0_m = Gamma_rmTS - epsilon hate_rmmin \n\nWe then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee that the obtained vectors have lower energy than the initial vector Γmin\n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Tuple. The returned paramaters are as follows => Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownfromTS-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, Vector{T}, Int64}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.rollDownfromTS","text":"rollDownTS(qaoa::QAOA, Γmin::Vector{T}, ig::Int; ϵ=0.001, tsType=\"symmetric\")\n\nStarting from a local minima we construct a vector corresponding to the transition state specified by ig. From there we construct two new vectors \n\nGamma^0_p = Gamma_rmTS + epsilon hate_rmmin\n\nGamma^0_m = Gamma_rmTS - epsilon hate_rmmin \n\nWe then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee that the obtained vectors have lower energy than the initial vector Γmin\n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\nig::Int: Index of the γ component where the zeros are inserted. \ntsType=\"symmetric\": In this case, the index of the β component is equal to ig. Otherwise, the β index is ig-1.\noptim=Val(:BFGS): Means that we will use the L-BFGS algorithm to perform the optimization. The other option is optim=Val{:GD}.\n\nReturn\n\nresult:Tuple. The returned paramaters are as follows => Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.spinChain-Tuple{Int64}","page":"API","title":"QAOALandscapes.spinChain","text":"spinChain(n::Int; bcond=\"pbc\")\n\nConstructs the graph for the classical Ising Hamiltonian on a chain with periodic boundary conditions determined by the keyword argument bcond\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.toFundamentalRegion!-Union{Tuple{T}, Tuple{M}, Tuple{H}, Tuple{P}, Tuple{QAOA{P, H, M}, AbstractVector{T}}} where {P, H, M, T<:Real}","page":"API","title":"QAOALandscapes.toFundamentalRegion!","text":"toFundamentalRegion!(qaoa::QAOA, Γ::Vector{Float64})\n\nImplements the symmetries of the QAOA for different graphs. For more detail see the following reference.\n\nFor an arbitrary graph, we can restrict both gamma and beta parameters to the -pi2 pi2 interval. Furthermore, beta parameters can be restricted even further to the -pi4 pi4 interval (see here) Finally, when dealing with regular graphs with odd degree \\gamma paramaters can be brought to the -pi4 pi4 interval. This function modifies inplace the initial input vector Γ. \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.transitionState-Union{Tuple{T}, Tuple{Vector{T}, Int64}} where T<:Real","page":"API","title":"QAOALandscapes.transitionState","text":"transitionState(Γp::Vector{Float64}, i::Int; tsType='symmetric')\n\nGiven an initial state Γp::Vector{Float64} of length 2p it creates another vector ΓTs of size 2p+2 such that the i-th γ component of ΓTs is 0 and the i-th (i-1-th)  β component of ΓTs is zero if tsType='symmetric' (tsType='non_symmetric') while all the other components are the same as Γp\n\nKeyword arguments\n\ntsType='symmetric' Only strings values 'symmetric' and 'non_symmetric' are accepted\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.transitionState-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Real","page":"API","title":"QAOALandscapes.transitionState","text":"transitionState(Γp::Vector{Float64})\n\nGiven an initial state Γp::Vector{Float64} of length 2p it creates a matrix M_rmTS of size 2p+2 times 2p+1.  The columns of M_rmTS corresponds to the transition states associated with the initial minimum Γp. The first p+1  columns correspond to symmetric TS while the remaining p columns correspond to non-symmetric TS.\n\n\n\n\n\n","category":"method"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"CurrentModule = QAOALandscapes","category":"page"},{"location":"#QAOALandscapes","page":"QAOALandscapes","title":"QAOALandscapes","text":"","category":"section"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"Documentation for QAOALandscapes.","category":"page"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"","category":"page"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"Modules = [QAOALandscapes]","category":"page"}]
}
