module QAOALandscapes

export QAOA, Hx, Hz, spinChain, constructUnitary, addCircuitLayer!, costFunction, gradCostFunction, hessianCostFunction, elementHessianCostFunction, getQAOAState
export interpInitialization, transitionState, toFundamentalRegion!, train!, permuteHessian, index1Direction
export rollDownInterp, rollDownTS, greedySelect, greedyOptimize, interpOptimize, rollDownWithCurvature, greedyOptimizeWithCurvature
export getHessianEigval
export gradStdTest, selectSmoothParameter, whichTSType, _onehot
export fourierJacobian, fourierInitialization, fromFourierParams, toFourierParams, trainFourier!, fourierOptimize, rollDownFourier
export fwht, ifwht
export getStateJacobian, quantumFisherInfoMatrix, getInitParameter
export spinChain

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