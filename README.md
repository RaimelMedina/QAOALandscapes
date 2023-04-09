# QAOALandscapes

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://raimelmedina.github.io/QAOALandscapes/dev/)

This repository contains the code for the QAOA landscape exploration project. Currently, we have implemented three different initialization and optimization strategies:

- Transition states
- Interpolation (Interp) strategy
- Fourier strategy

In terms of optimization, we currently support two gradient-based methods: 
- Gradient Descent using the Adam optimizer
- BFGS with BackTracking order 3. 

We use [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The computation of the cost function/energy gradient is done analytically. 

This a work in progress and the code is very very rudimentary. I hope this would be in a decent state to share sometime in the future. 