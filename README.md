# QAOALandscapes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/dev/)


QAOALandscapes is a Julia package for simulating the QAOA algorithm for MaxCut type problems with the goal of understanding and exploring the classical optimization landscape. We would like to understand why some of the heuristic initialization/optimization strategies out there work the way they do. Currently, three initialization/optimization strategies are implemented:

- Transition states.
- Interpolation (Interp) strategy.
- Fourier strategy.

In terms of optimization, we currently support all the methods available through [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The computation of the cost function/energy gradient is done using the **adjoint differentiation method** from this very nice paper [*Efficient calculation of gradients in classical simulations of variational quantum algorithms*](https://arxiv.org/abs/2009.02823).

## Installation

To install QAOALandscapes, open the Julia REPL and run the following command:

```julia
] activate QAOALandscapes
] instantiate
```
## Usage 
Here is an example of how to use QAOALandscapes to solve a MaxCut type problem:

```julia
using QAOALandscapes
using Graphs, Random

n    = 10
d    = 3  # for 3-regular random graphs
pmax = 10 # maximum circuit depth to explore
g    = random_regular_graph(n, d)

qaoa = QAOA(n, g, applySymmetries = true) # Uses the parity symmetry of the problem
_, init_point, init_energy = getInitialParameter(qaoa, spacing = 0.01) # obtain initial parameters at p=1

# Now choose a strategy
# For example, for transition states we have implemented the Greedy strategy
greedyData = greedyOptimize(qaoa, init_point, pmax);
```

## Warning
This is a work in progress and the code is very very rudimentary. I hope in this would be in a decent state to share sometime in the future. 