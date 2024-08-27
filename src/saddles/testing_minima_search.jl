using QAOALandscapes
using LinearAlgebra
using ProgressMeter
using Random
using Graphs
using Plots
using QAOAHomology
using Ripserer
import TensorCrossInterpolation as TCI
using Distances
using Optim
using LaTeXStrings
default(fontfamily="Computer Modern", framestyle=:box, grid=true, tickfontsize=12, guidefont=12, legendfontsize=12)
Base.Threads.nthreads()

mycolors = ["#49997c", "#1ebecd", "#ae3918", "#027ab0", "#d19c2f"];


Random.seed!(123)
g = random_regular_graph(10, 3);
problem = ClassicalProblem(Float64, g)
qaoa = QAOA(problem)

# optimization graph #
Γinit, Einit = getInitialParameter(qaoa)

function warmOptimizeAdam(qaoa::QAOA, p::Int, npoints::Int, niters::Int=200)
    vector_of_Γs = rand(2p, npoints) * 2π
    
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x]) = QAOALandscapes.findMinimaAdam(qaoa, vector_of_Γs[:, x], niters)
        next!(prog)
    end
    return vector_of_params, vector_of_energs
end

function modulatedNewton(qaoa::QAOA{P, H, M}, Γ0::Vector{T}, niter::Int=400) where {P<:AbstractProblem, H<:AbstractVector, M<:AbstractMixer, T<:AbstractFloat}
    g!(G, x) = G .= gradCostFunction(qaoa, x)
    h!(H, x) = H .= hessianCostFunction(qaoa, x, diffMode=:manual)

    results = Optim.optimize(qaoa, g!, h!, Γ0, method=Optim.Newton(), iterations = niter)
    parameters = Optim.minimizer(results)
    energies = Optim.minimum(results)
    toFundamentalRegion!(qaoa, parameters)
    return parameters, energies
end

function warmOptimizeBFGS(qaoa::QAOA, p::Int, npoints::Int, niters::Int=1000)
    vector_of_Γs = rand(2p, npoints) * 2π
    
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x]) = modulatedNewton(qaoa, vector_of_Γs[:, x], niters)
        next!(prog)
    end
    return vector_of_params, vector_of_energs
end

function warmOptimizeIPNewton(qaoa::QAOA, p::Int, npoints::Int, bounds::Tuple{V, V}) where V
    vector_of_Γs = rand(2p, npoints) * 2π
    
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x]) = QAOALandscapes.optimizeIPNewton(qaoa, vector_of_Γs[:, x], bounds)
        next!(prog)
    end
    return vector_of_params, vector_of_energs
end

function warmOptimizeFminbox(qaoa::QAOA, p::Int, npoints::Int, bounds::Tuple{V, V}) where V
    vector_of_Γs = rand(2p, npoints) * 2π
    
    vector_of_params = similar(vector_of_Γs)
    vector_of_energs = zeros(npoints)
    
    prog = Progress(npoints)
    Threads.@threads for x ∈ axes(vector_of_Γs, 2)
        (vector_of_params[:, x], vector_of_energs[x]) = QAOALandscapes.optimizeIPNewton(qaoa, vector_of_Γs[:, x], bounds)
        next!(prog)
    end
    return vector_of_params, vector_of_energs
end


lb = [0., -π/4, -π/4, -π/4];
ub = [π/4, π/4, π/4, π/4];

Γ_bfgs, E_bfgs = warmOptimizeBFGS(qaoa, 2, 10_000);

Γ_ipnewton, E_ipnewton = warmOptimizeIPNewton(qaoa, 2, 10_000, tuple(lb, ub));

Γ_fminbox, E_fminbox = warmOptimizeFminbox(qaoa, 2, 10_000, tuple(lb, ub));


weights_hist(iters::Int) = fill(1.0/iters, iters)

xlim_all = (
    1.1*min(minimum(E_bfgs), minimum(E_ipnewton), minimum(E_fminbox)), 
    3.)

plot_bfgs = Plots.histogram(
    E_bfgs, 
    bins = 400, 
    weights=weights_hist(10_000), 
    label="BFGS",
    yscale=:log10,
    legend=:topright,
    xlims = xlim_all
    )

plot_ipnewton = Plots.histogram(
    E_ipnewton, 
    bins = 400, 
    weights=weights_hist(10_000), 
    label="IPNewton",
    yscale=:log10,
    legend=:topright,
    xlims=xlim_all,
    color=mycolors[1]
    )

plot_fminbox = Plots.histogram(
    E_fminbox, 
    bins = 400, 
    weights=weights_hist(10_000), 
    label="Fminbox",
    yscale=:log10,
    legend=:topright,
    xlims=xlim_all,
    color=mycolors[1]
)


Plots.plot(plot_bfgs, plot_ipnewton, plot_fminbox, layout=(3, 1), size=(800, 600))


# NOW COMES THE TT-cross
p = 2
ϵ = 0.0
γ1 = (0., π/4)
γn = (-π/4, π/4)
tolerance = 1e-6

grid_spec = GridSpecifications(p, 70, γ1, γn);
pgrid = parameter_grid(grid_spec);

@time tci, ranks, errors = tci_energy_grid(qaoa, pgrid, 10; verbosity = 2, tolerance=tolerance)

TCI.linkdims(tci)

@time homology_data = Homology0Data(p, 0., tci, pgrid, filtration_type=:Rips);
nintervals = length(homology_data.h0)

cc = birth.(homology_data.h0)
persistent_components = findall(x->x>0.01, persistence.(homology_data.h0))

plot_homology = Plots.histogram(
    cc[persistent_components], 
    bins = 250, 
    weights=weights_hist(length(cc[persistent_components])),
    xlims=xlim_all,
    label=L"$H_0$ of components with persistence above $0.01$",
    xlabel="Energy"
    )


full_plot = Plots.plot(plot_bfgs, plot_ipnewton, plot_fminbox, plot_homology, 
    layout=(4, 1), size=(800, 900)
    )
Plots.savefig(full_plot, "box_constraint_vs_homology_001.pdf")

    plot_homology = Plots.histogram(
    birth.(homology_data.h0), bins=400, 
    weights=fill(1/nintervals, nintervals), 
    label=L"H_0",
    yscale=:log10);



    plot_homology
birth.(homology_data.h0) |> minimum

index_to_grid(x::Vector{Int}) = map(i->pgrid.iterators[i][x[i]], eachindex(x))
index_to_grid(x::CartesianIndex{N}) where N = index_to_grid(collect(x.I))

index_h0_components = vertices.(birth_simplex.(homology_data.h0));
params_from_homology = map(x->index_to_grid(x[1]), index_h0_components) 

good_homology_params = findall(x->x<1e-5, 
    map(norm, map(x->gradCostFunction(qaoa, x), params_from_homology)))

bad_homology_params = setdiff(1:nintervals, good_homology_params)

data_optim_homology = map(x->optimizeParameters(qaoa, x), params_from_homology)
energ_optim_homology = map(x->getindex(x, 2), data_optim_homology)
params_optim_homology = map(x->getindex(x, 1), data_optim_homology)
Plots.histogram(energ_optim_homology, bins=200, label="Optimizing from H0 components")

plot_norm_homology = Plots.scatter(
    map(norm, map(x->gradCostFunction(qaoa, x), params_from_homology[bad_homology_params])), 
    yscale=:log10,
    label=L"Gradient norm $H_0$ components")


minimum(energs_p_2)

Plots.plot(plot_optim, plot_homology, plot_norm_homology, layout=(3, 1))