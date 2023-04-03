<!-- ```@meta
CurrentModule = QAOALandscapes
```

# QAOALandscapes

Documentation for [QAOALandscapes](https://github.com/RaimelMedina/QAOALandscapes.jl).

```@index
```

```@autodocs
Modules = [QAOALandscapes]
``` -->
# QAOALandscapes.jl

This package implements universal function approximators (UFAs) with the quantum
simulator [`Yao.jl`](https://github.com/QuantumBFS/Yao.jl).

For example, the type `QNN` implements a universal approximator with inputs $x$
and variational parameters $\theta$
```math
f(x,\theta) = <\mathcal U(x,\theta)^\dagger| C |\mathcal U(x,\theta)>.
```
Derivatives of `QNN`s can be accessed via `dfdx`, `d2fdx2`, `d2fdxdtheta`, etc.
Derivatives w.r.t. inputs are currently implemented via finite differencing due to problems
with higher order derivatives. This will be fixed as soon as
`Diffractor.jl` is production ready.


For `QNN`s we also have a (potentiall slightly faster) legacy manual differentiation mode
which you can use by constructing `QNN`s with `QNN{ManualDiff}(args...)`

!!! warning
    The manual derivative functions (e.g. `dfdx`) assume that only one
    featuremap per feature is in the chain of circuits. Data reupload
    is therefore currently not supported.

!!! warning
    Mixed third order derivatives are currently not implemented. `d3fdx3` can
    only be called with same derivative indices, e.g.: `derivative_indices =
    (2,2,2)`



## Universal Function Approximators

This package currently implements two kinds of universal function approximators (UFAs):
1. Quantum Neural Networks (`QNN`): [1D Tutorial](@ref fitting_qnn)
2. Kernel UFAs (`KernelUFA`): [1D Tutorial](@ref fitting_kernel_ufa)

For details, you can check out the docstrings of the respective types which can
be accessed via `?name` (e.g.  for the type `QNN` type `?QNN` in the repl) or
take a look at the [`examples`](https://github.com/qu-co/QuantumNeuralNetworks.jl/tree/main/examples)

To run the examples start julia in the `examples` folder with `julia --project`
to activate the environment that contains all necessary dependencies.  Then run
`]instantiate` to install them. After they downloaded you can stop the
compilation of all packages with `ctrl-c` if you want to avoid compiling
dependencies you may not need (e.g. the diffeq related dependencies can take a
while). When you run an example with `include` only the needed dependencies will
by compiled.


## Type Hiearchy

The type hiearchy of this package is designed around `QuantumLayer`s which can
be composed to create `QuantumChain`s and ultimately things like a `QNN`.  Every
`QuantumLayer` is assumed to implement a call method that accepts inputs `x`
(either as a `AbstractVector` or as `AbstractMatrix` for batching), and a
`AbstractVector` of variational parameters.

For example, `FeatureMap`s and `VariationalAnsatz`s are `QuantumLayer`s:
```@repl block
using QuantumNeuralNetworks
FeatureMap(3) isa QuantumNeuralNetworks.QuantumLayer
VariationalAnsatz(3,3) isa QuantumNeuralNetworks.QuantumLayer
```

Every `QuantumLayer` has to implement a call method that accepts both inputs and
parameters (it does not necessarily have to use both) __and returns an
`AbstractBlock`__. E.g. a `FeatureMap` will only use the supplied inputs:
```julia
# call method
(fm::FeatureMap)(x::AbstractVector, θ::AbstractVector) = ...
```
and return a featuremap circuit
```@repl block
fm = FeatureMap(2);
fm([0.1], [])
```
With this interface it becomes possible to implement the `QuantumChain`
```math
\mathcal U(x,\theta) = \mathcal U_{\text{VC}}(\theta) \, \mathcal U_{\text{FM}}(x)
```
which can work with any newly defined `QuantumLayer` that can (possibly) act on both
inputs and parameters.
```@repl block
using Yao
fm = FeatureMap(5)
vc = VariationalAnsatz(5,5)
ch = QuantumChain(fm, vc)
ch(zero_state(5), [0.5], parameters(vc))  # U(θ)|U(x)|0>
```

With `QuantumChain`s and a cost function we can build a QNN which can also be
called with inputs and parameters to compute the forward pass:
```@example block
using QuantumNeuralNetworks, Yao

qnn = QNN(
    total_magnetization(5),
    QuantumChain(
        FeatureMap(5),
        VariationalAnsatz(5,5)))

# forward pass with batch of 1d inputs and initial parameters
qnn(rand(1,5), parameters(qnn))
```

Check the tutorials section for descriptions on how to define your own
[featuremaps](@ref featuremaps) and [variational circuits](@ref ansatze)


## Multi-Threading

You can run batches of inputs in parallel via a multi-threaded map (provided by `ThreadsX.jl`).
The only thing you have to do is start julia with the desired number of threads `N`:

```
julia --threads N

# or shorter
julia -t N
```

If you work with an IDE e.g. VSCode you can set an environment variable in the
shell that is used by your IDE as described
[here](https://docs.julialang.org/en/v1/manual/multi-threading/).  For Linux you
can add this to your shell's config file (e.g. `.bashrc`):
```bash
export JULIA_NUM_THREADS=4
```
In Atom you can set the number of threads in the Julia option as described
[here](https://docs.junolab.org/v0.6/man/settings.html#Julia-Options-1).


## Installation

Download and install [Julia](https://julialang.org/downloads/) and launch the
REPL.  This package is not on the official julia registry, so you have to
install it via the github repo.  If you have set up your SSH keys such that you
can clone a repo from Github you can enter the package REPL via `]` and do the
following
```julia
pkg> add git@github.com:qu-co/QuantumNeuralNetworks.jl.git
```
which should install the latest released version. If this does not work, clone
the repository and install it via
```julia
pkg> add /path/to/QuantumNeuralNetworks.jl/
```
and check out the latest released version by choosing a git tag:
```bash
git tag  # list all tags
git checkout v<tag-nr>
```
If you install the package this way you will have to manually update it by
pulling and choosing a new tag.