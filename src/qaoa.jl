@doc raw"""
    QAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))

Constructor for the `QAOA` object.
"""
mutable struct QAOA{T1 <: AbstractGraph}
    N::Int
    graph::T1
    HB::AbstractVector{Float64}
    HC::AbstractVector{Float64}
end

function QAOA(N::Int, g::T; applySymmetries=true) where T <: AbstractGraph
    if applySymmetries==false
        QAOA{T}(N, g, HxDiag(g), HzzDiag(g))
    else
        QAOA{T}(N-1, g, HxDiagSymmetric(g), HzzDiagSymmetric(g)) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N), and
    graph = $(qaoa.graph)"
    print(io,str)
end

@doc raw"""
    HxDiagSymmetric(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `HxDiagSymmetric` would be a vector of size ``2^{N-1}``. This construction, only makes sense if the cost/problem 
Hamiltonian ``H_C`` is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HxDiagSymmetric(g::T) where T<: AbstractGraph
    N = nv(g)
    Hx_vec = zeros(ComplexF64, 2^(N-1))
    count = 0
    for j ∈ 0:2^(N-1)-1
        if parity_of_integer(j)==0
            count += 1
            for i ∈ 0:N-1
                Hx_vec[count] += ComplexF64(-2 * (((j >> (i-1)) & 1) ) + 1)
            end
            Hx_vec[2^(N-1) - count + 1] = - Hx_vec[count]
        end
    end
    return Hx_vec
end

@doc raw"""
    HzzDiagSymmetric(g::T) where T <: AbstractGraph

Construct the cost Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `HzzDiagSymmetric` would be a vector of size ``2^{N-1}``. This construction, only makes sense if the cost/problem 
Hamiltonian ``H_C`` is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HzzDiagSymmetric(g::T) where T <: AbstractGraph
    N = nv(g)
    matZZ = zeros(ComplexF64, 2^(N-1));
    for edge ∈ edges(g)
        for j ∈ 0:2^(N-1)-1
            matZZ[j+1] += ComplexF64(-2 * (((j >> (edge.src -1)) & 1) ⊻ ((j >> (edge.dst -1)) & 1)) + 1) * getWeight(edge)
        end
    end
    return matZZ
end

#####################################################################

function getElementMaxCutHam(x::Int, graph::T) where T <: AbstractGraph
    val = 0.
    for i ∈ edges(graph)
        i_elem = ((x>>(i.src-1))&1)
        j_elem = ((x>>(i.dst-1))&1)
        idx = i_elem ⊻ j_elem
        val += ((-1)^idx)*getWeight(i)
    end
    return val
end

@doc raw"""
    HzzDiag(g::T) where T <: AbstractGraph

Construct the cost Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is ``N``, the corresponding Hamiltonian would be a vector of size ``2^{N-1}``.
This function instead returs a vector of size ``2^N``. 
"""
function HzzDiag(g::T) where T <: AbstractGraph
    result = ThreadsX.map(x->getElementMaxCutHam(x, g), 0:2^N-1)
	return result/2
end

function getElementMixingHam(x::Int, graph::T) where T <: AbstractGraph
    val = 0.
    N   = nv(graph)
    for i=1:N
        i_elem = ((x>>(i-1))&1)
        val += (-1)^i_elem
    end
    return val
end

@doc raw"""
    HzzDiag(g::T) where T<: AbstractGraph

Construct the mixing Hamiltonian. If the cost Hamiltonian is invariant under the parity operator
``\prod_{i=1}^N \sigma^x_i`` it is better to work in the +1 parity sector of the Hilbert space since
this is more efficient. In practice, if the system size is $N$, the corresponding Hamiltonianwould be a vector of size ``2^{N-1}``.
"""
function HxDiag(g::T) where T <: AbstractGraph
    result = ThreadsX.map(x->getElementMixingHam(x, g), 0:2^N-1)
	return result
end
#####################################################################

@doc raw"""
    getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T

Construct the QAOA state. More specifically, it returns the state:

```math
    |\Gamma^p \rangle = U(\Gamma^p) |+\rangle
```
with
```math
    U(\Gamma^p) = \prod_{l=1}^p e^{-i H_{B} \beta_{2l}} e^{-i H_{C} \gamma_{2l-1}}
```
and ``H_B, H_C`` corresponding to the mixing and cost Hamiltonian correspondingly.
"""
function getQAOAState(q::QAOA, Γ::T) where T <: AbstractVector
    p = length(Γ) ÷ 2
    
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]

    # First set the state vector to |+⟩
    ψ = 2^(-q.N/2)*ones(ComplexF64, 2^q.N)

    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        ψ .= fwht(ψ)              # Fast Hadamard transformation
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ)             # inverse Fast Hadamard transformation
    end
    return ψ
end

function getQAOAState(q::QAOA, Γ::T, ψ0::AbstractVector{ComplexF64}) where T <: AbstractVector
    p = length(Γ) ÷ 2
    
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]
    
    # First set the state vector to |+⟩
    ψ = copy(ψ0)

    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        ψ .= fwht(ψ)              # Fast Hadamard transformation
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ)             # inverse Fast Hadamard transformation
    end
    return ψ
end

@doc raw"""
    (q::QAOA)(Γ::T) where T <: AbstractVector

Computes the expectation value of the cost function ``H_C`` in the ``|\Gamma^p \rangle`` state. 
More specifically, it returns the following real number:

```math
    E(\Gamma^p) = \langle \Gamma^p |H_C|\Gamma^p \rangle
```
"""
function (q::QAOA)(Γ::T) where T <: AbstractVector
    ψ = getQAOAState(q, Γ)
    return real(ψ' * (q.HC .* ψ))
end

function ∂βψ(q::QAOA, Γ::T, layer::Int) where T <: AbstractVector
    p = length(Γ) ÷ 2
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]
    
    ψ = 2^(-q.N/2)*ones(ComplexF64, 2^q.N)
    
    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        ψ .= fwht(ψ) # Fast Hadamard transformation
        if i==layer
            ψ .= (q.HB .* ψ)
        end
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ) # inverse Fast Hadamard transformation
    end
    # multiply by -i 
    ψ .*= -im
    return ψ
end

function ∂γψ(q::QAOA, Γ::T, layer::Int) where T <: AbstractVector
    p = length(Γ) ÷ 2
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]

    ψ = state(uniform_state(q.N))[:]
    
    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        if i==layer 
            ψ .= (q.HC .* ψ)
        end
        ψ .= fwht(ψ)
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ) # inverse Fast Hadamard transformation
    end
    # multiply by -i 
    ψ .*= -im
    return ψ
end

@doc raw"""
    gradCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T

Computes the cost function gradient at the point ``\Gamma`` in parameter space, that is
```math
    \partial_l E(\Gamma^p) = \partial_l (\langle \Gamma^p |)H_C|\Gamma^p \rangle + \langle \Gamma^p |H_C \partial_l(|\Gamma^p \rangle)
```
"""
function gradCostFunction(qaoa::QAOA, Γ::T) where T <: AbstractVector
    p = length(Γ) ÷ 2
    γ = 1:2:2p
    β = 2:2:2p

    ψ = getQAOAState(qaoa, Γ)
    gradVector = zeros(eltype(Γ), length(Γ))

    for i ∈ 1:p
        gradVector[γ[i]]  = 2.0*real(∂γψ(qaoa, Γ, i)' * (qaoa.HC .* ψ))
        gradVector[β[i]]  = 2.0*real(∂βψ(qaoa, Γ, i)' * (qaoa.HC .* ψ))
    end
    return gradVector
end

@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. At the moment, we do it by using the [`FiniteDiff.jl`](https://github.com/JuliaDiff/FiniteDiff.jl)
"""
function hessianCostFunction(q::QAOA, θ::T) where T <: AbstractVector
    f(θ) = q(θ)
    matHessian = FiniteDiff.finite_difference_hessian(f, θ)
    return matHessian
end

@doc raw"""
    optimizeParameters(::Val{:GD}, qaoa::QAOA, Γ0::Vector{Float64}; niter=2000, tol=1e-5, printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `ADAM`
optimizer. By default the number of iterations is set to be `niter=2000`. The optimization stops whenever
the maximum number of iterations `niter` is reached or if the gradient norm is below the tolerance `tol=1e-5`
value.

# Arguments

* `Val(:GD)`: For using the gradient descent algorithm. Alternatively, `Val(:BFGS)` for using the `LBFGS` algorithm
* `qaoa::QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `niter::Int=2000`: Maximum number of iterations permitted
* `tol::Float64=1e-5`: Tolerance for the gradient norm
* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `params::Vector{Float64}`: Optimal parameter obtained
* `energy_history::Vector{Float64}`: Vector of size `niter` containing the values of the energy after each optimization stage
"""
function optimizeParameters(::Val{:GD}, qaoa::QAOA, Γ0::Vector{Float64}; niter=2000, tol=1e-5, printout=false)
    params = copy(Γ0)
    (printout) && println("Begining optimization using ADAM optimizer")
    energy_history = Float64[]

    for i in 1:niter
        ∇E = gradCostFunction(qaoa, params)
        if norm(∇E) < tol
            (printout) && println("convergence reached with |∇E|=$(norm(∇E))")
            break
        else
            E  = qaoa(params) 
            push!(energy_history, E) 
            Flux.Optimise.update!(ADAM(), params, ∇E)
        end
    end
    toFundamentalRegion!(qaoa, params)
    return params, energy_history[end]
end

@doc raw"""
    optimizeParameters(::Val{:BFGS}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `BFGS`
optimizer. 

# Arguments

* `Val(:BFGS)`: For using the BFGS. Alternatively, `Val(:GD)` for using the `ADAM` optimizer
* `qaoa:QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `parameters::Vector{Float64}`: Optimal parameter obtained
* `cost::Float64`: Value of the cost function for the optimal parameter obtained.
"""
function optimizeParameters(::Val{:BFGS}, qaoa::QAOA, params::Vector{Float64}; printout=false)
    
    (printout) && println("Begining optimization using BFGS optimizer")
    f(x::Vector{Float64}) = qaoa(x)
    function ∇f!(G, x::Vector{Float64})
        G .= gradCostFunction(qaoa, x)
    end
    algo = Optim.BFGS(linesearch = BackTracking(order=3))
    result = Optim.optimize(f, ∇f!, params, method=algo)

    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)

    toFundamentalRegion!(qaoa, parameters)
    if printout
        gradientNorm = norm(gradCostFunction(qaoa, parameters))
        print("Optimization with final cost function value E=$(cost), and gradient norm |∇E|=$(gradientNorm)")
    end
    return parameters, cost
end


@doc raw"""
    optimizeParameters(::Val{:Fourier}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `BFGS`
optimizer. Here we use the alternative "Fourier" initialization, where instead of optimizing the usual (γ, β) parameters
we optimize their frecuency components ``(u_{\gamma}, u_{\beta})``.

# Arguments

* `Val(:BFGS)`: For using the BFGS. Alternatively, `Val(:GD)` for using the `ADAM` optimizer
* `qaoa:QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `parameters::Vector{Float64}`: Optimal parameter obtained
* `cost::Float64`: Value of the cost function for the optimal parameter obtained.
"""
function optimizeParameters(::Val{:Fourier}, qaoa::QAOA, params::Vector{Float64}; printout=false)
    (printout) && println("Begining optimization using BFGS optimizer")

    f(x::Vector{Float64})  = qaoa(fromFourierParams(x))
    function ∇f!(G, x::Vector{Float64}) 
        G .= gradCostFunctionFourier(qaoa, x)
    end
    algo = Optim.BFGS(linesearch = BackTracking(order=3))
    result = Optim.optimize(f, ∇f!, params, method=algo)
    
    parameters = Optim.minimizer(result)
    cost       = Optim.minimum(result)

    if printout
        gradientNorm = gradCostFunctionFourier(qaoa, parameters) |> norm
        print("Optimization with final cost function value E=$(cost), and gradient norm |∇E|=$(gradientNorm)")
    end

    return fromFourierParams(parameters), cost
end

@doc raw"""
    getStateJacobian(q::QAOA, θ::T) where T <: AbstractVector

Returns the jacobian ``\nabla |\psi\rangle \in M(\mathbb{C}, 2^N \times 2p)``, where ``N`` corresponds to the total 
number of qubits and ``2p`` is the number of paramaters in ``|\psi(\theta)\rangle``. 
"""
function getStateJacobian(q::QAOA, θ::T) where T <: AbstractVector
    f(x) = getQAOAState(q, x)
    matJacobian = FiniteDiff.finite_difference_jacobian(f, ComplexF64.(θ))
    return matJacobian
end


@doc raw"""
    quantumFisherInfoMatrix(q::QAOA, θ::Vector{Float64})

Constructs the Quantum Fisher Information matrix, defined as follows

``
\mathcal{F}_{ij} = 4 \mathop{\rm{Re}}[\langle \partial_i \psi| \partial_j \psi\rangle - \langle \partial_i \psi| \psi\rangle \langle \psi|\partial_j \psi\rangle ]
``
"""
function quantumFisherInfoMatrix(q::QAOA, θ::T) where T <: AbstractVector
    ∇ψ = getStateJacobian(q, θ)
    ψ  = getQAOAState(q, θ)
    w  = ψ' * ∇ψ

    fisherMat = ∇ψ' * ∇ψ - kron(w', w)
    
    return 4.0*real(fisherMat)
end

@doc raw"""
    getInitParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)

Given a `QAOA` object it performs a grid search on a region of the two dimensional space spanned by ``\{ \gamma_1, \beta_1\}``
The ``\beta_1`` component is in the interval ``[-\pi/4, \pi/4]`` while the ``\gamma_1`` part is in the ``(0, \pi/4]`` for 3RRG
or ``(0, \pi/2]`` for dRRG (with ``d\neq 3``). 

We then launch the `QAOA` optimization procedure from the point in the 2-dimensional grid with the smallest cost function value.

# Returns
* 3-Tuple containing: 1.) the cost function grid, 2.) the optimal parameter, and 3.) the optimal energy
"""
function getInitParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)
    δ = spacing
    βIndex  = collect(-π/4:δ:π/4)
    
    if  isdRegularGraph(qaoa.graph, 3)
        γIndex  = collect(0.0:(spacing/2):π/4)
    else
        δ = 0.03
        γIndex  = collect(-π/2:δ:π/2)
    end
    
    energy  = zeros(length(γIndex), length(βIndex))
    for j in eachindex(βIndex)
        for i in eachindex(γIndex)
            energy[i,j] = qaoa([γIndex[i], βIndex[j]])
        end
    end
    pos = argmin(energy)
    Γ   = [γIndex[pos[1]], βIndex[pos[1]]] 

    gradNormGridMin = norm(gradCostFunction(qaoa, Γ))
    if gradNormGridMin > gradTol
        newParams, newEnergy = optimizeParameters(Val(:BFGS), qaoa, Γ)
        println("Convergence reached. Energy = $(newEnergy), |∇E| = $(norm(gradCostFunction(qaoa, newParams)))")
    end
    return energy, newParams, newEnergy
end