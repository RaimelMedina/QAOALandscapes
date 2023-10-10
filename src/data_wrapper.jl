mutable struct Parameter{T<:Real} <: AbstractVector{T}
    value::T
    data::Vector{T}
end

Base.getindex(p::Parameter, i::Int) = p.data[i]
Base.size(p::Parameter) = size(p.data)
Base.length(p::Parameter) = length(p.data)
Base.setindex!(p::Parameter{T}, v::T, i::Int) where T<:Real = (p.data[i]=v)

function setvalue!(param::Parameter{T}, qaoa::QAOA) where T<:Real
    param.value = qaoa(param)
end
function setvalue!(param::Parameter{T}, val::T) where T<:Real
    param.value = val
end

Parameter(vec::Vector{T}) where T<:Real = Parameter(T(0), vec)

(qaoa::QAOA)(param::Parameter{T}) where T<:Real = qaoa(param.data)


struct QAOAData{G<:AbstractGraph, T<:Real}
    graph::G
    init_param::Vector{T}
    gs_energy::T
    gs_indices::Vector{Int}
    greedy_data::AbstractVector{Parameter{T}}
    fourier_data::AbstractVector{Parameter{T}}
end

function QAOAData(g::G, pmax::Int, R::Int; seed=123) where G<:AbstractGraph
    setRandomSeed(seed)
    qaoa    = QAOA(g)

    @info "Collecting initial parameters"
    @time Γ0, E0 = getInitialParameter(qaoa, spacing = 0.01);
    
    gs_energ, gs_states = getSmallestEigenvalues(qaoa)
    

    @info "---Starting collecting optimization data---"
    greedy   = greedyOptimize(qaoa, Γ0, pmax, 1) 
    @info "Finished collecting greedy data"
    
    fourier  = fourierOptimize(qaoa, Γ0, pmax, R)
    @info "Finished collecting fourier data"

    greedy_params  = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    fourier_params = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    
    for i ∈ 1:pmax
        greedy_params[i]  = Parameter(greedy[i][2])
        setvalue!(greedy_params[i], greedy[i][1])

        fourier_params[i] = Parameter(fourier[i][2])
        setvalue!(fourier_params[i], fourier[i][1])
    end
    return QAOAData(g, Γ0, Float64(gs_energ), gs_states, greedy_params, fourier_params)
end