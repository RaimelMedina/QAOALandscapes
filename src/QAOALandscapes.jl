module QAOALandscapes

# Functions related to an arbitrary QAOA  
export QAOA, HxDiag, HxDiagSymmetric, HzzDiag, HzzDiagSymmetric, gradCostFunction, hessianCostFunction, getQAOAState, elementHessianCostFunction, optimizeParameters
export toFundamentalRegion!
export getStateJacobian, quantumFisherInfoMatrix
export getInitParameter
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

using Yao
using Graphs
using FiniteDiff
using Flux
using SimpleWeightedGraphs
using Optim
using LineSearches
using LinearAlgebra
using FLoops
using ThreadsX
using Statistics

include("qaoa.jl")
include("hessian_tools.jl")
include("transition_states.jl")
include("greedy_ts.jl")
include("interp.jl")
include("fourier.jl")
include("utils.jl")


end
