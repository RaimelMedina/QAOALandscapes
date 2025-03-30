module QAOALandscapes

const MAX_THREADS = 1024

# Functions related to an arbitrary QAOA
export ClassicalProblem, hamiltonian, XMixer, AbstractProblem, AbstractMixer 
export QAOA, getQAOAState, gradCostFunction, hessianCostFunction, geometricTensor
export rollDown, optimizeParametersSlice, optimizeWithStrategy
export plus_state, getInitialParameter, toFundamentalRegion!
# Functions related to different initialization strategies
# Interp
export InterpInitialization
# Fourier
# export toFourierParams, fromFourierParams, fourierInitialization, fourierJacobian, rollDownFourier, fourierOptimize
# Transition states
export TSInitialization, permuteHessian, getNegativeHessianEigval, getNegativeHessianEigvec, rollDownfromTS, rollDownTS, greedyOptimize, greedySelect, getHessianIndex
# General stationary points
export getStationaryPoints, gradSquaredNorm, optimizeGradSquaredNorm, gad
export modulatedNewton, warmOptimizeModulatedNewton
export QAOAData
export Node, IdNodes, constructOptimizationGraph
export TaylorTermsTS, Oϵ_ψ0, ψT2, ψT4, ψHC2

# Some useful Functions
export goemansWilliamson
# Benchmark with respect to Harvard hard harvard instance
export harvardGraph
export labs_hamiltonian

export xorsat_dict

abstract type AbstractQAOACost end
abstract type QuantumCost <: AbstractQAOACost end
abstract type ClassicalCost <: AbstractQAOACost end

# Problem and Mixer types 
abstract type AbstractProblem end
abstract type AbstractMixer end

# Measurement method hierarchy
abstract type ExpectationMethod end
struct ExactMethod <: ExpectationMethod end
struct SamplingMethod <: ExpectationMethod
    nshots::Int
    
    function SamplingMethod(nshots::Int)
        nshots > 0 || throw(ArgumentError("Number of shots must be positive"))
        new(nshots)
    end
end

abstract type AbstractInitialization end

struct InterpInitialization <: AbstractInitialization end
struct TSInitialization{T<:Real} <: AbstractInitialization
    ϵ::T
    TSInitialization(tol::T = T(1/1000)) where T<:Real = new{T}(tol)
end
struct FourierInitialization{T<:Real} <: AbstractInitialization
    R::Int
    α::T
end

struct TSResult{T<:Real, S<:Val}
    params::Matrix{T}
    energies::Vector{T}
    index::Int
    ts_type::S
    
    function TSResult(
        opt_params::Matrix{T},
        opt_energies::Vector{T},
        idx::Int,
        ts_type::S
    ) where {T<:Real, S<:Val}
        @assert size(opt_params, 2) == 2 && length(opt_energies) == 2 "Matrix is expected to have 2 columns coming from optimizing a TS along the +/- unique descend direction"
        @assert ts_type ∈ [Val(:symmetric), Val(:non_symmetric)] "ts_type must be either :symmetric or :non_symmetric"
        @assert idx > 0 "Index must be positive"
        return new{T, S}(
            opt_params,
            opt_energies,
            idx,
            ts_type
        )
    end

    # Undef constructor
    function TSResult{T,S}(::UndefInitializer) where {T<:Real, S<:Val}
        params = Matrix{T}(undef, 0, 2)
        energies = Vector{T}(undef, 0)
        # Create an instance of the Val type
        return new{T,S}(params, energies, 1, S())
    end

    # Convenience constructor
    function TSResult{T}(::UndefInitializer) where {T<:Real}
        # Create an instance of Val{:symmetric}
        return TSResult{T,Val{:symmetric}}(undef)
    end
end


using Revise
using GPUArrays
using Metal
using SparseArrays
using Graphs
using ForwardDiff
using Random
using AbstractTrees
using ProgressMeter
using SimpleWeightedGraphs
using Optimization
using LineSearches
using LinearAlgebra
using ThreadsX
using Statistics
using FiniteDiff
using StatsBase
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
# include(joinpath("base", "metal.jl"))


# inside /classical
include(joinpath("classical", "maxcut.jl"))

# inside /experimental
# include(joinpath("experimental", "data_wrapper.jl"))
include(joinpath("experimental", "experimental.jl"))

# inside /initializations
# include(joinpath("initializations", "fourier.jl"))
include(joinpath("initializations", "interp.jl"))
include(joinpath("initializations", "greedy_ts.jl"))
include(joinpath("initializations", "transition_states.jl"))
include(joinpath("initializations", "hessian_tools.jl"))

# inside /saddles
include(joinpath("saddles", "saddles_search.jl"))

# inside /utilities
# include(joinpath("utilities", "utils.jl"))
# include(joinpath("utilities", "state_utilities.jl"))
include("test_instances.jl")

end
