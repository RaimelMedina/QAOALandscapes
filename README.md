# QAOALandscapes

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RaimelMedina.github.io/QAOALandscapes.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RaimelMedina.github.io/QAOALandscapes.jl/dev/)
[![Build Status](https://github.com/RaimelMedina/QAOALandscapes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RaimelMedina/QAOALandscapes.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/RaimelMedina/QAOALandscapes.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/RaimelMedina/QAOALandscapes.jl)

This repository implements contains the code for the QAOA landscape exploration project. Currently, we have implemented three different initialization and optimization strategies:
    
    1. Transition states
    2. Interpolation (Interp) strategy
    3. Fourier strategy

In terms of optimization, we currently support Gradient Descent with *Adam* optimizer and *BFGS* with BackTracking order 3. We use [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/)

This a work in progress and the code is very very rudimentary. I hope this would be in a decent state to share sometime in the future. 