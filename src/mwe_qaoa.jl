using Optim
using NLopt
using LinearAlgebra
using Graphs
using SciPy
using BenchmarkTools
using Random
using Yao
using LineSearches

rng = MersenneTwister(123);


#### Code for the Fast Walsh-Hadamard transform #####
function fwht(a::Vector{T}) where T
    h   = 1
    dim = length(a) 
    tmp = copy(a)
    while 2 * h <= dim
        for i ∈ 0:2h:dim-2
            for j ∈ i:(i + h - 1)
                x = tmp[j + 1]
                y = tmp[j + h + 1]
                tmp[j + 1] = x + y
                tmp[j + h + 1] = x - y
            end
        end
        h *= 2
    end
    return tmp
end

function ifwht(a::Vector{T}) where T
    return fwht(a) / length(a)
end

#### Code for the QAOA object ######
mutable struct QAOA 
    N::Int
    graph::Graph
    HB::Vector{ComplexF64}
    HC::Vector{ComplexF64}
end
QAOA(N::Int, g::Graph) = QAOA(N, g, Hx_diag(g), Hzz_diag(g))

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N), and
    graph = $(qaoa.graph)"
    print(io,str)
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

function Hzz_diag(g::Graph{T}) where T
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

function Hx_diag(g::Graph{T}) where T
    N = nv(g)
	result = ThreadsX.map(x->getElementMixingHam(x, N), 0:2^N-1)
	return result
end

function getQAOAState(q::QAOA, Γ::Vector{Float64})
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

function (q::QAOA)(Γ::Vector)
    ψ = getQAOAState(q, Γ)
    return real(ψ' * (q.HC .* ψ))
end

function ∂βψ(q::QAOA, Γ::Vector, layer::Int)
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
        ψ .= ifwht(ψ)             # inverse Fast Hadamard transformation
    end
    return -im*ψ
end

function ∂γψ(q::QAOA, Γ::Vector, layer::Int)
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
        ψ .= ifwht(ψ)             # inverse Fast Hadamard transformation
    end
    return -im*ψ
end

function gradCostFunction(qaoa::QAOA, Γ::Vector)
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

#### Define a particular QAOA instance #####

N      = 10
g      = random_regular_graph(N, 3, rng=rng)
qaoa   = QAOA(N, g);

#### Initial point at p=16 ####

Γ0 = [0.0004999999999999999, -0.0007071067811865476, 0.06147383110266019, -0.6251485096555397, 0.1532844298161166, -0.503522331285764, 
0.2046291857751561, -0.4437970820577217, 0.24282296812332146, -0.4079792119464043, 0.26596804060484386, -0.382406157742222, 0.2829548006964458, 
-0.367699301255467, 0.294080524534996, -0.35659539316501254, 0.3034167838430033, -0.34457551725346414, 0.3164874767999871, -0.32563587119317283, 
0.33651834024550803, -0.29617732546483144, 0.36676991143769583, -0.25451676824074454, 0.40316991030528126, -0.20492680571570787, 
0.45257334630403767, -0.1521157871183456, 0.4989607778241654, -0.10735838415189591, 0.5219352105707442, -0.04622435828710203]

function train_scipy(qaoa::QAOA, Γ0::Vector{Float64})
    f(x::Vector{Float64})  = qaoa(x)
    ∇f(x::Vector{Float64}) = gradCostFunction(qaoa, x)

    result = SciPy.optimize.minimize(f, Γ0, jac=∇f, method="BFGS")
    return result
end

function train_optim(qaoa::QAOA, Γ0::Vector{Float64})
    f(x::Vector{Float64}) = qaoa(x)
    function g!(G, x::Vector{Float64})
        G .= gradCostFunction(qaoa, x)
    end
    algo = Optim.BFGS(;alphaguess = LineSearches.InitialStatic(), linesearch = LineSearches.MoreThuente())
    result = Optim.optimize(f, g!, Γ0, method=algo)
    return result
end

function train_nlopt(qaoa::QAOA, Γ0::Vector{Float64})
    
    function f(x::Vector{Float64}, grad::Vector{Float64})
        if length(grad) > 0
            grad .= gradCostFunction(qaoa, x)
        end
        qaoa(x)
    end
    opt = Opt(:LD_LBFGS, length(Γ0))
    min_objective!(opt, f)
    cost, parameters, info = NLopt.optimize(opt, Γ0)
end

@btime data_scipy = train_scipy(qaoa, Γ0);
@btime data_optim = train_optim(qaoa, Γ0);
@btime data_nlopt = train_nlopt(qaoa, Γ0);

