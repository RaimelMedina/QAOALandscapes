module QAOALandscapes
using SparseArrays
const OperatorType{T} = Union{SparseMatrixCSC{T, Int}, Vector{T}}

# Functions related to an arbitrary QAOA  
export QAOA, HxDiag, HxDiagSymmetric, HzzDiag, HzzDiagSymmetric, generalClassicalHamiltonian,  getQAOAState, gradCostFunction, hessianCostFunction, geometricTensor
export HessianCostFunction, optimizeParameters, optimizeParametersSlice
export plus_state, getInitialParameter, toFundamentalRegion!

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
using ProgressMeter
using SimpleWeightedGraphs
using Optim
using LineSearches
using LinearAlgebra
using Printf
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

# inside /general/
include(joinpath("base", "qaoa.jl"))
include(joinpath("base", "gradient.jl"))
include(joinpath("base", "hamiltonians.jl"))
include(joinpath("base", "layers.jl"))
include(joinpath("base", "optimization_settings.jl"))
include(joinpath("base", "parameters.jl"))

# inside /classical
include(joinpath("classical", "maxcut.jl"))

# inside /experimental
include(joinpath("experimental", "data_wrapper.jl"))
include(joinpath("experimental", "experimental.jl"))

# inside /initializations
include(joinpath("initializations", "fourier.jl"))
include(joinpath("initializations", "interp.jl"))
include(joinpath("initializations", "greedy_ts.jl"))
include(joinpath("initializations", "transition_states.jl"))
include(joinpath("initializations", "hessian_tools.jl"))

# inside /saddles
include(joinpath("saddles", "saddles_search.jl"))

# inside /utilities
include(joinpath("utilities", "utils.jl"))
include(joinpath("utilities", "state_utilities.jl"))

include("harvard_instance.jl")
end
