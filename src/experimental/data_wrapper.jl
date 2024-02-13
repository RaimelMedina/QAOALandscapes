struct QAOAData{G<:AbstractGraph, T<:Real, B<:AbstractBackend}
    graph::G
    init_param::Vector{T}
    gs_energy::T
    gs_indices::Vector{Int}
    # interp_data::AbstractVector{Parameter{T}}
    fourier_data::AbstractVector{Parameter{T}}
end

function QAOAData(T::Type{<:Real}, g::G, pmax::Int; seed=123) where G<:AbstractGraph
    setRandomSeed(seed)
    qaoa    = QAOA(T, g)

    @info "Collecting initial parameters"
    @time Γ0, E0 = getInitialParameter(qaoa);
    
    @show E0
    stateEquivC = getEquivalentClasses(qaoa.HC |> real, rounding=true)
    gs_energ, gs_states = qaoa.HC[stateEquivC[1][1]] |> real, stateEquivC[1]
    

    @info "---Starting collecting optimization data---"
    # interp_data   = interpOptimize(qaoa, Γ0, pmax, 1) 
    # @info "Finished collecting interp data"
    
    fourier  = interpOptimize(qaoa, Γ0, pmax)
    @info "Finished collecting fourier data"

    # greedy_params  = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    fourier_params = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    
    for i ∈ 1:pmax
        
        fourier_params[i] = Parameter(fourier[i][2])
        setvalue!(fourier_params[i], fourier[i][1])
    end
    return QAOAData{typeof(g), T, CPUBackend}(g, Γ0, gs_energ, gs_states, fourier_params)
end

function QAOAData(B::Type{<:METALBackend}, T::Type{<:Real}, g::G, pmax::Int; seed=123) where G<:AbstractGraph
    setRandomSeed(seed)
    qaoa    = QAOA(B, T, g)

    @info "Collecting initial parameters"
    @time Γ0, E0 = getInitialParameter(qaoa)
    
    @show E0
    stateEquivC = getEquivalentClasses(qaoa.HC |> Array |> real, rounding=true)
    gs_energ, gs_states = Array(qaoa.HC)[stateEquivC[1][1]] |> real, stateEquivC[1]
    

    @info "---Starting collecting optimization data---"
    # interp_data   = interpOptimize(qaoa, Γ0, pmax, 1) 
    # @info "Finished collecting interp data"
    
    fourier  = interpOptimize(qaoa, Γ0, pmax)
    @info "Finished collecting fourier data"

    # greedy_params  = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    fourier_params = Vector{Parameter{eltype(Γ0)}}(undef, pmax)
    
    for i ∈ 1:pmax
        # greedy_params[i]  = Parameter(greedy[i][2])
        # setvalue!(greedy_params[i], greedy[i][1])

        fourier_params[i] = Parameter(fourier[i][2])
        setvalue!(fourier_params[i], fourier[i][1])
    end
    return QAOAData{typeof(g), T, METALBackend}(g, Γ0, gs_energ, gs_states, fourier_params)
end