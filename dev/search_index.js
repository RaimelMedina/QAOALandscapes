var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [QAOALandscapes]\nOrder   = [:type, :function]","category":"page"},{"location":"api/#QAOALandscapes.QAOA","page":"API","title":"QAOALandscapes.QAOA","text":"    struct QAOA{T1 <: AbstractGraph, T2}\n        N::Int\n        graph::T1\n        HB::AbstractVector{T2}\n        HC::AbstractVector{T2}\n    end\n\nQAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))\n\nConstructor for the QAOA object.\n\n\n\n\n\n","category":"type"},{"location":"api/#QAOALandscapes.HxDiag-Tuple{T} where T<:Graphs.AbstractGraph","page":"API","title":"QAOALandscapes.HxDiag","text":"HzzDiag(g::T) where T<: AbstractGraph\n\nConstruct the mixing Hamiltonian. If the cost Hamiltonian is invariant under the parity operator prod_i=1^N sigma^x_i it is better to work in the +1 parity sector of the Hilbert space since this is more efficient. In practice, if the system size is N, the corresponding Hamiltonianwould be a vector of size 2^N-1.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.HxDiagSymmetric-Tuple{T} where T<:Graphs.AbstractGraph","page":"API","title":"QAOALandscapes.HxDiagSymmetric","text":"HxDiagSymmetric(g::T) where T<: AbstractGraph\n\nConstruct the mixing Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system  size is N, then HxDiagSymmetric would be a vector of size 2^N-1. This construction, only makes sense if the cost/problem  Hamiltonian H_C is invariant under the action of the parity operator, that is\n\n    H_C prod_i=1^N sigma^x_i = 0\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.HzzDiag-Tuple{T} where T<:Graphs.AbstractGraph","page":"API","title":"QAOALandscapes.HzzDiag","text":"HzzDiag(g::T) where T <: AbstractGraph\n\nConstruct the cost Hamiltonian. If the cost Hamiltonian is invariant under the parity operator prod_i=1^N sigma^x_i it is better to work in the +1 parity sector of the Hilbert space since this is more efficient. In practice, if the system size is N, the corresponding Hamiltonian would be a vector of size 2^N-1. This function instead returs a vector of size 2^N. \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.HzzDiagSymmetric-Tuple{T} where T<:Graphs.AbstractGraph","page":"API","title":"QAOALandscapes.HzzDiagSymmetric","text":"HzzDiagSymmetric(g::T) where T <: AbstractGraph\n\nConstruct the cost Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system  size is N, then HzzDiagSymmetric would be a vector of size 2^N-1. This construction, only makes sense if the cost/problem  Hamiltonian H_C is invariant under the action of the parity operator, that is\n\n    H_C prod_i=1^N sigma^x_i = 0\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.fourierOptimize-Tuple{QAOA, Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.fourierOptimize","text":"fourierOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int)\n\nStarting from a local minima Γ0 at p=1 it performs the Fourier optimization strategy until the circuit depth pmax is reached. By default the BFGS optimizer is used. \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓ0::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Dict. Dictionary with keys being keys \\in [1, pmax] and values being a Tuple{Float64, Vector{Float64}} of cost function value and corresponding parameter.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getInitParameter-Tuple{QAOA}","page":"API","title":"QAOALandscapes.getInitParameter","text":"getInitParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)\n\nGiven a QAOA object it performs a grid search on a region of the two dimensional space spanned by  gamma_1 beta_1 The beta_1 component is in the interval -pi4 pi4 while the gamma_1 part is in the (0 pi4 for 3RRG or (0 pi2 for dRRG (with dneq 3). \n\nWe then launch the QAOA optimization procedure from the point in the 2-dimensional grid with the smallest cost function value.\n\nReturns\n\n3-Tuple containing: 1.) the cost function grid, 2.) the optimal parameter, and 3.) the optimal energy\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getNegativeHessEigval-Tuple{QAOA, Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.getNegativeHessEigval","text":"getNegativeHessEigvalssianEigval(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType=\"symmetric\", ϵ=cbrt(eps(Float64)))\n\nComputes the approximation to the minimum (negative) eigenvalue of the Hessian at the TS obtained by padding with zeros the local minimum Γmin. The transition state is completely specified by the index of the γ component ig, and the  type of transition states (\"symmetric\" or \"non_symmetric\"). The cost of obtaining this approximate eigenvalue is basically the cost of computing two matrix elements of a Hessian.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getQAOAState-Union{Tuple{T}, Tuple{QAOA, T}} where T<:(AbstractVector)","page":"API","title":"QAOALandscapes.getQAOAState","text":"getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T\n\nConstruct the QAOA state. More specifically, it returns the state:\n\n    Gamma^p rangle = U(Gamma^p) +rangle\n\nwith\n\n    U(Gamma^p) = prod_l=1^p e^-i H_B beta_2l e^-i H_C gamma_2l-1\n\nand H_B H_C corresponding to the mixing and cost Hamiltonian correspondingly.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.getStateJacobian-Union{Tuple{T}, Tuple{QAOA, T}} where T<:(AbstractVector)","page":"API","title":"QAOALandscapes.getStateJacobian","text":"getStateJacobian(q::QAOA, θ::T) where T <: AbstractVector\n\nReturns the jacobian nabla psirangle in M(mathbbC 2^N times 2p), where N corresponds to the total  number of qubits and 2p is the number of paramaters in psi(theta)rangle. \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.gradCostFunction-Union{Tuple{T}, Tuple{QAOA, T}} where T<:(AbstractVector)","page":"API","title":"QAOALandscapes.gradCostFunction","text":"gradCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T\n\nComputes the cost function gradient at the point Gamma in parameter space, that is\n\n    partial_l E(Gamma^p) = partial_l (langle Gamma^p )H_CGamma^p rangle + langle Gamma^p H_C partial_l(Gamma^p rangle)\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.hessianCostFunction-Union{Tuple{T}, Tuple{QAOA, T}} where T<:(AbstractVector)","page":"API","title":"QAOALandscapes.hessianCostFunction","text":"hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T\n\nComputes the cost function Hessian at the point Gamma in parameter space. At the moment, we do it by using the FiniteDiff.jl\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.index1Direction-Tuple{QAOA, Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.index1Direction","text":"index1Direction(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; tsType=\"symmetric\", doChecks=false)\n\nComputes the approximate eigenvalue and index-1 eigenvector of the Hessian at the transition state obtained from the local minimum  Γmin. It is completely specified by the parameters iγ and tsType=\"symmetric\". If the optional parameter doChecks=false is set to true, then the function also returns the relative error in estimating the true eigenvalue as well as the inner product between the approximate and true eigenvector\n\nArguments\n\nqaoa::QAOA: QAOA object\nΓmin::Vector{Float64}: The vector corresponding to a local minimum of QAOAₚ. \nig::Int: Index of the γ component where the zeros are inserted. \ntsType=\"symmetric\": In this case, the index of the β component is equal to ig. Otherwise, the β index is ig-1.\n\nOptional arguments\n\ndoChecks=false: In this case the function returns a dictionary with keys eigvec_approx and eigval_approx. If set to true it has additional keys => change_basis, eigvec_fidelity and eigval_error\n\nReturns\n\nresult::Dict Dictionary with the following keys: eigvec_approx, eigval_approx. If doChecks=true the following additional keys are available: change_basis, eigvec_fidelity and eigval_error\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.interpInitialization-Tuple{Vector{Float64}}","page":"API","title":"QAOALandscapes.interpInitialization","text":"interpInitialization(Γp::Vector{Float64})\n\nGiven an initial state Γp::Vector{Float64} of length 2p it creates another vector ΓInterp of size 2p+2 with gamma (beta) components given by the following expression\n\ngamma^i_p+1 = fraci-1p gamma^i-1_p + fracp-i+1pgamma^i_p\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.interpOptimize-Tuple{QAOA, Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.interpOptimize","text":"interpOptimize(qaoa::QAOA, Γ0::Vector{Float64}, pmax::Int; optim = Val(:BFGS))\n\nStarting from a local minima Γ0 at p=1 it performs the Interp optimization strategy until the circuit depth pmax is reached. By default the BFGS optimizer is used. \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓ0::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Dict. Dictionary with keys being keys \\in [1, pmax] and values being a Tuple{Float64, Vector{Float64}} of cost function value and corresponding parameter.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParameters-Tuple{Val{:BFGS}, QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.optimizeParameters","text":"optimizeParameters(::Val{:BFGS}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)\n\nPerform optimization of the QAOA using the gradient descent algorithm with the BFGS optimizer. \n\nArguments\n\nVal(:BFGS): For using the BFGS. Alternatively, Val(:GD) for using the ADAM optimizer\nqaoa:QAOA: QAOA object\nΓ0::Vector{Float64}: Initial point from where the optimization starts\n\nKeyword arguments\n\nprintout=false: Whether if we print something during the optimization or not\n\nReturns\n\nIt returns a tuple containing the following information\n\nparameters::Vector{Float64}: Optimal parameter obtained\ncost::Float64: Value of the cost function for the optimal parameter obtained.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParameters-Tuple{Val{:Fourier}, QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.optimizeParameters","text":"optimizeParameters(::Val{:Fourier}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)\n\nPerform optimization of the QAOA using the gradient descent algorithm with the BFGS optimizer. Here we use the alternative \"Fourier\" initialization, where instead of optimizing the usual (γ, β) parameters we optimize their frecuency components (u_gamma u_beta).\n\nArguments\n\nVal(:BFGS): For using the BFGS. Alternatively, Val(:GD) for using the ADAM optimizer\nqaoa:QAOA: QAOA object\nΓ0::Vector{Float64}: Initial point from where the optimization starts\n\nKeyword arguments\n\nprintout=false: Whether if we print something during the optimization or not\n\nReturns\n\nIt returns a tuple containing the following information\n\nparameters::Vector{Float64}: Optimal parameter obtained\ncost::Float64: Value of the cost function for the optimal parameter obtained.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.optimizeParameters-Tuple{Val{:GD}, QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.optimizeParameters","text":"optimizeParameters(::Val{:GD}, qaoa::QAOA, Γ0::Vector{Float64}; niter=2000, tol=1e-5, printout=false)\n\nPerform optimization of the QAOA using the gradient descent algorithm with the ADAM optimizer. By default the number of iterations is set to be niter=2000. The optimization stops whenever the maximum number of iterations niter is reached or if the gradient norm is below the tolerance tol=1e-5 value.\n\nArguments\n\nVal(:GD): For using the gradient descent algorithm. Alternatively, Val(:BFGS) for using the LBFGS algorithm\nqaoa::QAOA: QAOA object\nΓ0::Vector{Float64}: Initial point from where the optimization starts\n\nKeyword arguments\n\nniter::Int=2000: Maximum number of iterations permitted\ntol::Float64=1e-5: Tolerance for the gradient norm\nprintout=false: Whether if we print something during the optimization or not\n\nReturns\n\nIt returns a tuple containing the following information\n\nparams::Vector{Float64}: Optimal parameter obtained\nenergy_history::Vector{Float64}: Vector of size niter containing the values of the energy after each optimization stage\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.permuteHessian-Tuple{AbstractMatrix{Float64}, Int64}","page":"API","title":"QAOALandscapes.permuteHessian","text":"permuteHessian(H::AbstractArray{Float64,2}, i::Int; tsType=\"symmetric\")\n\nComputes the permutation that takes the Hessian at a particular transition state into the form described in the paper Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.\n\nArguments\n\nH::AbstractArray{Float64,2}: Hessian at the transition state in the original basis.\ni::Int: Index of the γ component at which the zero is added. If tsType='symmetric' then β=i, otherwise if tsType='non_symmetric' β=i-1.\n\nReturn\n\nlistOfIndices::Vector{Int64}: List of indices correponding to the arrangement of the new basis elements.\npermMat::Matrix{Float64}: Matrix implementing the desired permutation.\nHTransformed::Matrix{Float64}: Transformed Hessian at the transition state. Specifically, we have that H_mathoprmperm=PHP^-1.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.permuteHessian-Tuple{Int64, Int64}","page":"API","title":"QAOALandscapes.permuteHessian","text":"permuteHessian(depth::Int, i::Int; tsType=\"symmetric\")\n\nComputes the permutation that takes the Hessian at a particular transition state into the form described in the paper Basically, the last two rows and columns of the transformed Hessian correspond to the indexes where the zeros were inserted.\n\nArguments\n\ndepth::Int: Circuit depth of the transition state.\ni::Int: Index of the γ component at which the zero is added. If tsType='symmetric' then β=i, otherwise if tsType='non_symmetric' β=i-1.\n\nReturn\n\nlistOfIndices::Vector{Int64}: List of indices correponding to the arrangement of the new basis elements.\npermMat::Matrix{Float64}: Matrix implementing the desired permutation.\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.quantumFisherInfoMatrix-Union{Tuple{T}, Tuple{QAOA, T}} where T<:(AbstractVector)","page":"API","title":"QAOALandscapes.quantumFisherInfoMatrix","text":"quantumFisherInfoMatrix(q::QAOA, θ::Vector{Float64})\n\nConstructs the Quantum Fisher Information matrix, defined as follows\n\nmathcalF_ij = 4 mathoprmRelangle partial_i psi partial_j psirangle - langle partial_i psi psirangle langle psipartial_j psirangle \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownInterp-Tuple{QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.rollDownInterp","text":"rollDownInterp(qaoa::QAOA, Γmin::Vector{Float64}; optim = Val(:BFGS))\n\nStarting from a local minima we construct a new vector using the INTERP initialization. From there we carry out the optimization algorithm using the optim=Val(:BFGS) optimizer (otherwise optim=Val(:GD)) \n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Tuple. The first element corresponds to the vector corresponding to which the algorithm converged to, and the second element is correponding energy_history\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownTS-Tuple{QAOA, Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.rollDownTS","text":"rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}, ig::Int; ϵ=0.001, tsType=\"symmetric\")\n\nStarting from a local minima we construct a vector corresponding to the transition state specified by ig. From there we construct two new vectors \n\nGamma^0_p = Gamma_rmTS + epsilon hate_rmmin \n\nGamma^0_m = Gamma_rmTS - epsilon hate_rmmin \n\nWe then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee that the obtained vectors have lower energy than the initial vector Γmin\n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\nig::Int: Index of the γ component where the zeros are inserted. \ntsType=\"symmetric\": In this case, the index of the β component is equal to ig. Otherwise, the β index is ig-1.\noptim=Val(:BFGS): Means that we will use the L-BFGS algorithm to perform the optimization. The other option is optim=Val{:GD}.\n\nReturn\n\nresult:Tuple. The returned paramaters are as follows => Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.rollDownTS-Tuple{QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.rollDownTS","text":"rollDownTS(qaoa::QAOA, Γmin::Vector{Float64}; ϵ=0.001, optim=Val(:BFGS))\n\nStarting from a local minima we construct all transition states (a total of 2p+1 of them). From each of the transition states, we construct two new vectors \n\nGamma^0_p = Gamma_rmTS + epsilon hate_rmmin \n\nGamma^0_m = Gamma_rmTS - epsilon hate_rmmin \n\nWe then use these two vectors as initial points to carry out the optimization. Following our analytical results we are guarantee that the obtained vectors have lower energy than the initial vector Γmin\n\nArguments\n\nqaoa::QAOA: QAOA object \nΓmin::Vector{Float64}: Vector correponding to the local minimum from which we will construct the particular TS and then roll down from.\n\nReturn\n\nresult:Tuple. The returned paramaters are as follows => Γmin_m, Γmin_p, Emin_m, Emin_p, info_m, info_p\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.spinChain-Tuple{Int64}","page":"API","title":"QAOALandscapes.spinChain","text":"spinChain(n::Int; bcond=\"pbc\")\n\nConstructs the graph for the classical Ising Hamiltonian on a chain with periodic boundary conditions determined by the keyword argument bcond\n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.toFundamentalRegion!-Tuple{QAOA, Vector{Float64}}","page":"API","title":"QAOALandscapes.toFundamentalRegion!","text":"toFundamentalRegion!(Γ::Vector{Float64})\n\nImplements the symmetries of the QAOA for the case of the MaxCut problem on unweighted 3-regular graphs. More specifically, one gets that the resulting vector has its gamma_i beta_i components in the interval -pi4 pi4 forall i in p, except gamma_1 in 0 pi4. This function modifies inplace the initial input vector Γ. \n\n\n\n\n\n","category":"method"},{"location":"api/#QAOALandscapes.transitionState-Tuple{Vector{Float64}, Int64}","page":"API","title":"QAOALandscapes.transitionState","text":"transitionState(Γp::Vector{Float64}, i::Int; tsType='symmetric')\n\nGiven an initial state Γp::Vector{Float64} of length 2p it creates another vector ΓTs of size 2p+2 such that the i-th γ component of ΓTs is 0 and the i-th (i-1-th)  β component of ΓTs is zero if tsType='symmetric' (tsType='non_symmetric') while all the other components are the same as Γp\n\nKeyword arguments\n\ntsType='symmetric' Only strings values 'symmetric' and 'non_symmetric' are accepted\n\n\n\n\n\n","category":"method"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"CurrentModule = QAOALandscapes","category":"page"},{"location":"#QAOALandscapes","page":"QAOALandscapes","title":"QAOALandscapes","text":"","category":"section"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"Documentation for QAOALandscapes.","category":"page"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"","category":"page"},{"location":"","page":"QAOALandscapes","title":"QAOALandscapes","text":"Modules = [QAOALandscapes]","category":"page"}]
}
