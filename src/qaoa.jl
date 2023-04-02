@doc raw"""
    QAOA(N::Int, graph::Graph, HB::AbstractBlock, HC::AbstractBlock)
    QAOA(N::Int, graph::Graph; applySymmetries = true) = QAOA(N, graph, Hx_diag_symmetric(graph), Hzz_diag_symmetric(graph))

Constructor for the `QAOA` object.
"""
struct QAOA{T}
    N::Int
    graph::Graph
    HB::AbstractVector{T}
    HC::AbstractVector{T}
end

function QAOA(N::Int, g::Graph; applySymmetries=true)
    if applySymmetries==false
        QAOA(N, g, HxDiag(g), HzzDiag(g))
    else
        QAOA(N-1, g, HxDiagSymmetric(g), HzzDiagSymmetric(g)) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N), and
    graph = $(qaoa.graph)"
    print(io,str)
end

@doc raw"""
    HxDiagSymmetric(g::Graph{T})

Construct the mixing Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `Hx_diag_symmetric` would be a vector of size 2ᴺ⁻¹. This construction, only makes sense if the cost/problem 
Hamiltonian H₀ is invariant under the action of the parity operator, that is

```math
    [H_C, \prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HxDiagSymmetric(g::Graph{T}) where T
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
    Hzz_diag_symmetric(g::Graph{T})
    HzzDiagSymmetric(g::SimpleWeightedGraph{T})

Construct the cost Hamiltonian in the positive (+1) parity sector of the Hilbert space. This means that if the system 
size is N, then `Hz_diag_symmetric` would be a vector of size 2ᴺ⁻¹. This construction, only makes sense if the cost/problem 
Hamiltonian H₀ is invariant under the action of the parity operator, that is

```math
    [H_C, $\prod_{i=1}^N \sigma^x_i] = 0
```
"""
function HzzDiagSymmetric(g::Graph{T}) where T
    N = nv(g)
    edgeList = findall(!iszero, adjacency_matrix(g))
    matZZ = zeros(ComplexF64, 2^(N-1));
    for i ∈ edgeList
        for j ∈ 0:2^(N-1)-1
            matZZ[j+1] += ComplexF64(-2 * (((j >> (i[1] -1)) & 1) ⊻ ((j >> (i[2] -1)) & 1)) + 1)
        end
    end
    return matZZ/2
end

function HzzDiagSymmetric(g::SimpleWeightedGraph{T}) where T
    N = nv(g)
    edgeList = findall(!iszero, adjacency_matrix(g))
    matZZ = zeros(ComplexF64, 2^(N-1));
    for i ∈ edgeList
        for j ∈ 0:2^(N-1)-1
            matZZ[j+1] += ComplexF64(-2 * (((j >> (i[1] -1)) & 1) ⊻ ((j >> (i[2] -1)) & 1)) + 1) * get_weight(g, i[1], i[2])
        end
    end
    return matZZ/2
end

#####################################################################

function getElementMaxCutHam(x::Int, g::Vector{CartesianIndex{2}}, graph::SimpleWeightedGraph)
    val = 0.
    N = length(g)
    for i=1:N
        i_elem = ((x>>(g[i][1]-1))&1)
        j_elem = ((x>>(g[i][2]-1))&1)
        idx = i_elem ⊻ j_elem
        val += ((-1)^idx)*get_weight(graph, g[i][1], g[i][2])
    end
    return ComplexF64(val)
end

function getElementMaxCutHam(x::Int, g::Vector{CartesianIndex{2}})
    val = 0.
    N = length(g)
    for i=1:N
        i_elem = ((x>>(g[i][1]-1))&1)
        j_elem = ((x>>(g[i][2]-1))&1)
        idx = i_elem ⊻ j_elem
        val += ((-1)^idx)
    end
    return ComplexF64(val)
end

function HzzDiag(g::SimpleWeightedGraph{T}) where T
    interactionIndices = findall(!iszero, adjacency_matrix(g))
    N = nv(g)
	result = ThreadsX.map(x->getElementMaxCutHam(x, interactionIndices, g), 0:2^N-1)
	return result/2
end

function HzzDiag(g::Graph{T}) where T
    interactionIndices = findall(!iszero, adjacency_matrix(g))
    N = nv(g)
	result = ThreadsX.map(x->getElementMaxCutHam(x, interactionIndices), 0:2^N-1)
	return result/2
end

function getElementMixingHam(x::Int, N::Int)
    val = 0.
    for i=1:N
        i_elem = ((x>>(i-1))&1)
        val += (-1)^i_elem
    end
    return ComplexF64(val)
end

function HxDiag(g::Graph{T}) where T
    N = nv(g)
	result = ThreadsX.map(x->getElementMixingHam(x, N), 0:2^N-1)
	return result
end
#####################################################################

function getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T
    p = length(Γ) ÷ 2
    
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]

    ψ = state(uniform_state(q.N))[:]
    
    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        ψ .= fwht(ψ)              # Fast Hadamard transformation
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ)             # inverse Fast Hadamard transformation
    end
    return ψ
end

function (q::QAOA)(Γ::AbstractVector{T}) where T
    ψ = getQAOAState(q, Γ)
    return real(ψ' * (q.HC .* ψ))
end

function ∂βψ(q::QAOA, Γ::AbstractVector{T}, layer::Int) where T
    p = length(Γ) ÷ 2
    γ = Γ[1:2:2p]
    β = Γ[2:2:2p]
    
    ψ = state(uniform_state(q.N))[:]
    
    for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        ψ .= fwht(ψ) # Fast Hadamard transformation
        if i==layer
            ψ .= (q.HB .* ψ)
        end
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ψ .= ifwht(ψ) # inverse Fast Hadamard transformation
    end
    return -im*ψ
end

function ∂γψ(q::QAOA, Γ::AbstractVector{T}, layer::Int) where T
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
    return -im*ψ
end

function gradCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T
    p = length(Γ) ÷ 2
    γ = 1:2:2p
    β = 2:2:2p

    ψ = getQAOAState(qaoa, Γ)
    
    gradVector = zeros(Float64, length(Γ))
    for i ∈ 1:p
        gradVector[γ[i]]  = 2.0*real(∂γψ(qaoa, Γ, i)' * (qaoa.HC .* ψ))
        gradVector[β[i]]  = 2.0*real(∂βψ(qaoa, Γ, i)' * (qaoa.HC .* ψ))
    end
    return gradVector
end

function hessianCostFunction(q::QAOA, θ::AbstractVector{T}) where T
    f(θ) = q(θ)
    matHessian = FiniteDiff.finite_difference_hessian(f, θ)
    return matHessian
end

@doc raw"""
    train!(::Val{:GD}, qaoa::QAOA, Γ0::Vector{Float64}; niter=2000, tol=1e-5, printout=false)

Perform optimization of the `fQAOA` using the gradient descent algorithm with the `ADAM`
optimizer. By default the number of iterations is set to be `niter=1000`. The optimization stops whether
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
function train!(::Val{:GD}, qaoa::QAOA, Γ0::Vector{Float64}; niter=2000, tol=1e-5, printout=false)
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
    train!(::Val{:BFGS}, qaoa::QAOA, Γ0::Vector{Float64}; tol=1e-5, printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `L-BFGS`
optimizer. 

# Arguments

* `Val(:BFGS)`: For using the L-BFGS. Alternatively, `Val(:GD)` for using the `ADAM` optimizer
* `qaoa:QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `parameters::Vector{Float64}`: Optimal parameter obtained
* `cost::Float64`: Value of the cost function for the optimal parameter obtained.
* `info::Symbol`: Usually we get `:SUCCESS` if the optimization was successful. Otherwise, the message is different depending on the problem encountered by the algorithm
"""
function train!(::Val{:BFGS}, qaoa::QAOA, params::Vector{Float64}; printout=false)
    
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
    train!(::Val{:Fourier}, qaoa::QAOA, Γ0::Vector{Float64}; printout=false)

Perform optimization of the `QAOA` using the gradient descent algorithm with the `L-BFGS`
optimizer. Here we use the alternative "Fourier" initialization, where instead of optimizing the usual (γ, β) parameters
we optimize their frecuency components (uγ, uβ).

# Arguments

* `Val(:BFGS)`: For using the L-BFGS. Alternatively, `Val(:GD)` for using the `ADAM` optimizer
* `qaoa:QAOA`: QAOA object
* `Γ0::Vector{Float64}`: Initial point from where the optimization starts

# Keyword arguments

* `printout=false`: Whether if we print something during the optimization or not

# Returns
It returns a tuple containing the following information
* `parameters::Vector{Float64}`: Optimal parameter obtained
* `cost::Float64`: Value of the cost function for the optimal parameter obtained.
* `info::Symbol`: Usually we get `:SUCCESS` if the optimization was successful. Otherwise, the message is different depending on the problem encountered by the algorithm
"""
function train!(::Val{:Fourier}, qaoa::QAOA, params::Vector{Float64}; printout=false)
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
    getStateJacobian(q::QAOA, θ::Vector)

Returns the jacobian ``\nabla |\psi\rangle \in M(\mathbb{C}, 2^N \times 2p)``, where ``N`` corresponds to the total 
number of qubits and ``2p`` is the number of paramaters in ``|\psi(\theta)\rangle``. 
"""
function getStateJacobian(q::QAOA, θ::Vector{Float64})
    f(x::Vector) = getQAOAState(q, x)
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
function quantumFisherInfoMatrix(q::QAOA, θ::Vector{Float64})
    ∇ψ = getStateJacobian(q, θ)
    ψ  = getQAOAState(q, θ)
    w  = ψ' * ∇ψ

    fisherMat = ∇ψ' * ∇ψ - kron(w', w)
    
    return 4.0*real(fisherMat)
end

function getInitParameter(qaoa::QAOA; spacing = 0.01, gradTol = 1e-6)
    δ = spacing
    βIndex  = collect(-π/4:δ:π/4)
    
    if VQALandscapes.isdRegularGraph(qaoa.graph, 3)
        γIndex  = collect(0.0:(spacing/2):π/4)
    else
        δ = 0.03
        γIndex  = collect(0.0:δ:π/4)
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
        newParams, newEnergy = train!(Val(:BFGS), qaoa, Γ)
        println("Convergence reached. Energy = $(newEnergy), |∇E| = $(norm(gradCostFunction(qaoa, newParams)))")
    end
    return energy, newParams, newEnergy
end