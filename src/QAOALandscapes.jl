module QAOALandscapes
using SparseArrays
const OperatorType{T} = Union{SparseMatrixCSC{T,Int64}, Vector{T}}

# Functions related to an arbitrary QAOA  
export QAOA, HxDiag, HxDiagSymmetric, HzzDiag, HzzDiagSymmetric, generalClassicalHamiltonian,  getQAOAState, gradCostFunction, hessianCostFunction, geometricTensor
export elementHessianCostFunction, optimizeParameters

export toFundamentalRegion!

export getInitialParameter
# Functions related to different initialization strategies
# Interp
export interpInitialization, rollDownInterp, interpOptimize
# Fourier
export toFourierParams, fromFourierParams, fourierInitialization, fourierJacobian, rollDownFourier, fourierOptimize
# Transition states
export transitionState, permuteHessian, getNegativeHessianEigval, getNegativeHessianEigvec, rollDownfromTS, rollDownTS, greedyOptimize, greedySelect, getHessianIndex
# General stationary points
export getStationaryPoints, gradSquaredNorm, optimizeGradSquaredNorm, gad

export Node, IdNodes, constructPartialOptimizationGraph

# Some useful Functions
export spinChain
export gradStdTest, selectSmoothParameter, whichTSType, _onehot
export goemansWilliamson
# Benchmark with respect to Harvard hard harvard instance
export harvardGraph

using Revise
using Graphs
using ForwardDiff
using Random
using SimpleWeightedGraphs
using Optim
using LineSearches
using LinearAlgebra
using FLoops
using ThreadsX
using Statistics
using LoopVectorization 
using Distributions
using Base.Threads
using Combinatorics
using KrylovKit
using Convex
using SCS

function setRandomSeed(seed::Int)
    Random.seed!(seed)
end

include("qaoa.jl")
include("data_wrapper.jl")
include("layers.jl")
include("hamiltonians.jl")
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
include("saddles_search.jl")
include("experimental.jl")
include("maxcut.jl")
end
