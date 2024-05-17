module QAOALandscapes

const MAX_THREADS = 1024

# Functions related to an arbitrary QAOA
export ClassicalProblem, hamiltonian, XMixer, AbstractProblem, AbstractMixer 
export QAOA, getQAOAState, gradCostFunction, hessianCostFunction, geometricTensor
export optimizeParameters, optimizeParametersSlice, OptSetup
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

export QAOAData
export Node, IdNodes, constructOptimizationGraph
export TaylorTermsTS, Oϵ_ψ0, ψT2, ψT4, ψHC2

# Some useful Functions
export goemansWilliamson
# Benchmark with respect to Harvard hard harvard instance
export harvardGraph
export labs_hamiltonian

export xorsat_dict


abstract type AbstractProblem end
abstract type AbstractMixer end

# using Requires
# function __init__()
#     @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" include(joinpath("base", "gpu.jl"))
# end

using Revise
using GPUArrays
using SparseArrays
using Graphs
using ForwardDiff
using Random
using ProgressMeter
using SimpleWeightedGraphs
using Optim
using LineSearches
using LinearAlgebra
using ThreadsX
using Statistics
using Distributions
using Base.Threads
using Combinatorics
using Convex
using SCS



function setRandomSeed(seed::Int)
    Random.seed!(seed)
end

# inside /base/
include(joinpath("base", "problem.jl"))
include(joinpath("base", "x_mixer.jl"))
include(joinpath("base", "qaoa.jl"))
include(joinpath("base", "gradient.jl"))
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
include("test_instances.jl")

end
