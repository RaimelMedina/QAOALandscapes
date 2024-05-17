# QAOALandscapes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/dev/)


QAOALandscapes is a Julia package for exactly simulating the QAOA algorithm for general p-spin problems with the goal of understanding and exploring the classical optimization landscape. We would like to understand why some of the heuristic initialization/optimization strategies out there work the way they do. Currently, three initialization/optimization strategies are implemented:

- Transition states.
- Interpolation (Interp) strategy.
- Fourier strategy.

In terms of optimization, we currently support all the methods available through [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The computation of the cost function/energy gradient is done using the **adjoint differentiation method** from this very nice paper [*Efficient calculation of gradients in classical simulations of variational quantum algorithms*](https://arxiv.org/abs/2009.02823).

## Installation

To install QAOALandscapes, open the Julia REPL and run the following command:

```julia
] add https://github.com/RaimelMedina/QAOALandscapes.git
```
## Usage 
Here is an example of how to use QAOALandscapes to solve a MaxCut type problem:

```julia
using QAOALandscapes
using Graphs, Random

n    = 10
d    = 3  # for 3-regular random graphs
pmax = 10 # maximum circuit depth to explore
g    = random_regular_graph(n, d) # 3-regular unweighted graph
prob = ClassicalProblem(Float64, g)

qaoa = QAOA(prob) # if the problem is Z2 symmetric then the algorithm will work in the correct Hilbert subspace
init_point, init_energy = getInitialParameter(qaoa) # obtain initial parameters at p=1

# Now choose a strategy
# For example, for transition states we have implemented the Greedy strategy
greedyData = greedyOptimize(qaoa, init_point, pmax);

# If you prefer INTERP strategy
interpData = interpOptimize(qaoa, init_point, pmax);

# Alternatively, you can also optimize a given set of parameters
# directly
initial_parameter = rand(20)
optimal_param, optimal_energy = optimizeParameters(qaoa, initial_parameter)
```

# Documentation
Right now is not working but I'm working on that and it should be fixed soon.

# GPU
As of now, I don't know how to setup the package so that it automatically detects which GPU (if any) the user has. So for now, I recommend that after downloading the package you build it again by including the `gpu.jl` file and add the correspoding GPU package, i.e., `CUDA.jl` or `Metal.jl`. I plan to add support to AMD devices via `KernelAbstractions.jl` but that will come later

## Warning
This is a work in progress and the code is very very rudimentary. I hope in this would be in a decent state to share sometime in the future. 