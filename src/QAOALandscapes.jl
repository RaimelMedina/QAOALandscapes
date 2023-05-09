module QAOALandscapes

# Functions related to an arbitrary QAOA  
export QAOA, HxDiag, HxDiagSymmetric, HzzDiag, HzzDiagSymmetric, gradCostFunction, hessianCostFunction, getQAOAState, elementHessianCostFunction, optimizeParameters
export toFundamentalRegion!
export getStateJacobian, quantumFisherInfoMatrix
export getInitialParameter
# Functions related to different initialization strategies
# Interp
export interpInitialization, rollDownInterp, interpOptimize
# Fourier
export toFourierParams, fromFourierParams, fourierInitialization, fourierJacobian, rollDownFourier, fourierOptimize
# Transition states
export transitionState, permuteHessian, getNegativeHessianEigval, getNegativeHessianEigvec, rollDownTS, greedyOptimize, greedySelect

# Some useful Functions
export spinChain
export gradStdTest, selectSmoothParameter, whichTSType, _onehot

# Benchmark with respect to Harvard hard harvard instance
export harvardGraph

# Optimization settings
export OptimizationSettings

using DiffResults
using Graphs
using FiniteDiff
using ForwardDiff
using Flux
using SimpleWeightedGraphs
using Optim
using LineSearches
using LinearAlgebra
using FLoops
using ThreadsX
using Statistics
using LoopVectorization 

include("qaoa.jl")
include("hessian_tools.jl")
include("transition_states.jl")
include("greedy_ts.jl")
include("interp.jl")
include("fourier.jl")
include("utils.jl")
include("harvard_instance.jl")
include("optimization_settings.jl")
include("gradient_adjoint.jl")
include("state_utilities.jl")
end
