@doc raw"""
    QAOA(N::Int, graph::T; applySymmetries = true) where T<:AbstractGraph = QAOA{T, Float64}(N, graph, HxDiagSymmetric(graph), HzzDiagSymmetric(graph))

Constructor for the `QAOA` object.
"""
struct QAOA{T1 <: AbstractGraph, T2}
    N::Int
    graph::T1
    HB::AbstractVector{T2}
    HC::AbstractVector{T2}
    hamiltonian::OperatorType{T2}
    parity_symmetry::Bool
end

function QAOA(g::T; applySymmetries=true) where T <: AbstractGraph
    N = nv(g)
    if applySymmetries==false
        h = 2.0*HzzDiag(g)
        T2 = eltype(h)
        QAOA{T, eltype(h)}(N, g, HxDiag(g), h, h, false)
    else
        h = HzzDiagSymmetric(g)
        T2 = eltype(h)
        QAOA{T, eltype(h)}(N-1, g, HxDiagSymmetric(g), h, h, true) 
    end
end

function QAOA(g::T1, ham::OperatorType{T2}; applySymmetries=true) where {T1 <: AbstractGraph, T2}
    N = nv(g)
    if applySymmetries==false
        QAOA{T1, T2}(N, g, T2.(HxDiag(g)), T2.(HzzDiag(g)), ham, false)
    else
        QAOA{T1, T2}(N-1, g, T2.(HxDiagSymmetric(g)), T2.(HzzDiagSymmetric(g)), ham, true) 
    end
end

function Base.show(io::IO, qaoa::QAOA)
    str = "QAOA object with: 
    number of qubits = $(qaoa.N)."
    if qaoa.parity_symmetry
        str2 = "
    Z₂ parity symmetry"
        print(io, str * str2)
    else
        print(io, str)
    end
end

@doc raw"""
    getQAOAState(q::QAOA, Γ::AbstractVector{T}) where T <: Real

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
function getQAOAState(q::QAOA, Γ::AbstractVector{T}; use_fwht = false) where T <: Real
    p = length(Γ) ÷ 2
    ψ = 2^(-q.N/2)*ones(Complex{T}, 2^q.N)
    if use_fwht
        γ = @view Γ[1:2:2p]
        β = @view Γ[2:2:2p]
        @inbounds @simd for i ∈ 1:p
            ψ .= exp.(-im * (γ[i] .* q.HC)) .* ψ
            fwht!(ψ, q.N)              # Fast Hadamard transformation
            ψ .= exp.(-im * (β[i] .* q.HB)) .* ψ
            ifwht!(ψ, q.N)             # inverse Fast Hadamard transformation
        end
    else
        @inbounds @simd for i ∈ eachindex(Γ)
            applyQAOALayer!(q, Γ, i, ψ)
        end
    end
    return ψ
end

function getQAOAState(q::QAOA, Γ::AbstractVector{T}, ψ0::AbstractVector{Complex{T}}) where T <: Real
    p = length(Γ) ÷ 2
    
    γ = @view Γ[1:2:2p]
    β = @view Γ[2:2:2p]
    
    ψ = copy(ψ0)
    @inbounds @simd for i ∈ 1:p
        ψ .= exp.(-im * γ[i] * q.HC) .* ψ
        fwht!(ψ, q.N)              # Fast Hadamard transformation
        ψ .= exp.(-im * β[i] * q.HB) .* ψ
        ifwht!(ψ, q.N)             # inverse Fast Hadamard transformation
    end
    return ψ
end

@doc raw"""
    (q::QAOA)(Γ::AbstractVector{T}) where T <: Real

Computes the expectation value of the cost function ``H_C`` in the ``|\Gamma^p \rangle`` state. 
More specifically, it returns the following real number:

```math
    E(\Gamma^p) = \langle \Gamma^p |H_C|\Gamma^p \rangle
```
"""
function (q::QAOA)(Γ::AbstractVector{T}) where T <: Real
    ψ = getQAOAState(q, Γ)
    typeHam = typeof(q.hamiltonian)
    if typeHam <: Vector
        return real(ψ' * (q.hamiltonian .* ψ))
    else
        return real(dot(ψ, q.hamiltonian, ψ))
    end
end

function energyVariance(q::QAOA, Γ::AbstractVector{T}) where T<:Real
    h_mean_squared = q(Γ)^2
    ψ = getQAOAState(q, Γ)
    
    typeHam = typeof(q.hamiltonian)
    
    if typeHam <: Vector
        h_squared_mean = dot(ψ, ( q.hamiltonian .^2 ) .* ψ) |> real
    else
        h_squared_mean = dot(ψ, (q.hamiltonian^2), ψ) |> real
    end
    return h_squared_mean - h_mean_squared
end

@doc raw"""
    hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}) where T<:Real

Computes the cost function Hessian at the point ``\Gamma`` in parameter space. Currently, we use the [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl) package
"""
function hessianCostFunction(qaoa::QAOA, Γ::AbstractVector{T}; diffMode=:manual) where T<:Real
    if diffMode==:forward
        matHessian = ForwardDiff.hessian(qaoa, Γ)
        return matHessian
    elseif diffMode==:manual
        p = length(Γ) ÷ 2
        
        ψ = getQAOAState(qaoa, Γ)
        
        ψCol = similar(ψ)
        ψRow  = similar(ψ)
        ψRowCol  = similar(ψ)
        
        matHessian = zeros(2p, 2p)
        for col in 1:2p
            ψCol .= ∂ψ(qaoa, Γ, col)
            for row in col:2p
                ψRow .= ∂ψ(qaoa, Γ, row)
                ψRowCol .= ∂ψ(qaoa, Γ, col, row)
                
                matHessian[row, col] = 2*real(dot(ψRow, qaoa.HC .* ψCol)) + 2*real(dot(ψ, qaoa.HC .* ψRowCol))
                if col != row
                    matHessian[col, row] = matHessian[row, col]
                end
            end
        end
        return matHessian
    else
        throw(ArgumentError("diffMode=$(diffMode) not supported. Only ':manual' or ':forward' methods are implemented"))
    end
end

function ∂ψ(qaoa::QAOA, Γ::Vector{T}, i::Int) where T<:Real
    ψ = plus_state(T, qaoa.N)
    @inbounds @simd for idx ∈ eachindex(Γ)
        if idx==i
            applyQAOALayerDerivative!(qaoa, Γ, idx, ψ)
        else
            applyQAOALayer!(qaoa, Γ, idx, ψ)
        end
    end
    return ψ
end

function ∂ψ(qaoa::QAOA, Γ::Vector{T}, i::Int, j::Int) where T<:Real
    ψ = plus_state(T, qaoa.N)
    @inbounds @simd for idx ∈ eachindex(Γ)
        if i==j
            if idx==i
                applyQAOALayer!(qaoa, Γ, idx, ψ)
                if isodd(idx)
                    Hzz_ψ!(qaoa, ψ)
                    Hzz_ψ!(qaoa, ψ)
                    ψ .*= -1.0
                else
                    Hx_ψ!(qaoa, ψ)
                    Hx_ψ!(qaoa, ψ)
                    ψ .*= -1.0
                end
            else
                applyQAOALayer!(qaoa, Γ, idx, ψ)
            end
        else
            if idx==i || idx==j
                applyQAOALayerDerivative!(qaoa, Γ, idx, ψ)
            else
                applyQAOALayer!(qaoa, Γ, idx, ψ)
            end
        end
    end
    return ψ
end