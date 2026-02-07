struct QAOAData{G<:AbstractProblem, T<:Real}
    problem::G
    init_param::Vector{T}
    gs_energy::T
    gs_indices::Vector{Int}
    optim_data::AbstractVector{QAOALandscapes.Parameter{T}}
end

function QAOAData(T::Type{<:Real}, g::G, pmax::Int; seed=123) where G<:AbstractGraph
    QAOALandscapes.setRandomSeed(seed)
    if G <: SimpleWeightedGraph
        prob = ClassicalProblem(g)
    else
        prob = ClassicalProblem(T, g)
    end
    qaoa = QAOA(prob)

    @info "Collecting initial parameters"
    @time Γ0, E0 = getInitialParameter(qaoa);
    
    @show E0
    _, stateEquivC = QAOALandscapes.getEquivalentClasses(qaoa.HC |> real, rounding=false)
    gs_energ, gs_states = qaoa.HC[stateEquivC[1][1]] |> real, stateEquivC[1]
    
    @info "---- Ground state energy is E₀ = $(gs_energ) \n"

    @info "---Starting collecting optimization data---"
    
    local_minima  = fourierOptimize(qaoa, Γ0, pmax)
    @info "Finished collecting greedy-1 data"

    opt_params = Vector{QAOALandscapes.Parameter{eltype(Γ0)}}(undef, pmax)
    
    for i ∈ 1:pmax
        opt_params[i] = QAOALandscapes.Parameter(local_minima[i][2])
        QAOALandscapes.setvalue!(opt_params[i], local_minima[i][1])
    end
    return QAOAData{typeof(prob), T}(prob, Γ0, gs_energ, gs_states, opt_params)
end

function QAOAData(prob::ClassicalProblem{T}, pmax::Int; seed=123) where T<:Real
    QAOALandscapes.setRandomSeed(seed)
    qaoa = QAOA(prob)

    @info "Collecting initial parameters"
    @time Γ0, E0 = getInitialParameter(qaoa);
    
    @show E0
    _, stateEquivC = QAOALandscapes.getEquivalentClasses(qaoa.HC |> real, rounding=false)
    gs_energ, gs_states = qaoa.HC[stateEquivC[1][1]] |> real, stateEquivC[1]
    
    @info "---- Ground state energy is E₀ = $(gs_energ) \n"

    @info "---Starting collecting optimization data---"
    
    local_minima  = fourierOptimize(qaoa, Γ0, pmax)
    @info "Finished collecting greedy-1 data"

    opt_params = Vector{QAOALandscapes.Parameter{eltype(Γ0)}}(undef, pmax)
    
    for i ∈ 1:pmax
        opt_params[i] = QAOALandscapes.Parameter(local_minima[i][2])
        QAOALandscapes.setvalue!(opt_params[i], local_minima[i][1])
    end
    return QAOAData{typeof(prob), T}(prob, Γ0, gs_energ, gs_states, opt_params)
end