using QAOALandscapes
using Graphs
using HDF5
using Distributions
using Optim
using LinearAlgebra
using Plots

n = 10
g = path_graph(n)
add_edge!(g, n, 1)

prob = ClassicalProblem(Float64, g)
ham = hamiltonian(prob)
mixer = XMixer(n)
ψ  = fill(ComplexF64(1/sqrt(2^(prob.n-1))), 2^(prob.n-1))

T = typeof(ham)
M = typeof(mixer)
K = typeof(prob)

qaoa = QAOA{K, T, M, QAOALandscapes.QuantumCost}(n, prob, ham, mixer, ψ)

Γinit, Einit = getInitialParameter(qaoa)

qaoa(Γinit)

opt_graph = constructOptimizationGraph(qaoa, Γinit, 4);

gg, edges_optg, energies_optg, circdepth_optg = QAOALandscapes.getEdgesFromOptGraph(qaoa, opt_graph, construct_graph=true);

using GraphRecipes
GraphRecipes.graphplot(gg, curves=false, y = map(x->energies_optg[x], 1:nv(gg)))

using NetworkLayout, GraphMakie, CairoMakie

function custom_layout(graph)
    # Get the x coordinates from a standard layout algorithm
    xy_coords = NetworkLayout.spring(graph)
    
    # Combine with our fixed y coordinates
    return [[xy_coords[i][1], energies_optg[i]] for i in 1:nv(graph)]
end
GraphMakie.graphplot(gg, layout = custom_layout)

NetworkLayout.spring(gg)

edges_optg
energies_optg

# gs_index = qaoa.HC |> real |> findmin

Γts = transitionState(Γinit, 1)
gradCostFunction(qaoa, Γinit)
qaoa(Γts)  ≈ qaoa(Γinit)
gΓts = norm(gradCostFunction(qaoa, Γts))
hΓts_eigvals = eigvals(hessianCostFunction(qaoa, Γts))
u = getNegativeHessianEigvec(qaoa, Γinit, 1, doChecks=true)["eigvec_approx"] |> Array
xvals = range(-0.5, 0.5, length=100);
yvals = map(x->qaoa(Γts+u*x), xvals);

Plots.plot(xvals, yvals)
fourierInitialization(Γinit) |> qaoa
ΓallTS = sum(transitionState(Γinit), dims=2) ./ 2 |> vec
qaoa(ΓallTS)

energies_optg[circdepth_optg]

sopt_m, sopt_p = optimizeParametersSlice(qaoa, Γinit, 1) 
Optim.minimizer(sopt_m)

slice_optima["energy"].x_opt


h5open("tfim_opt_graph.h5", "w") do file
    write(file, "edges", edges_optg)
    write(file, "vertexW", energies_optg)
    write(file, "depth", circdepth_optg)
end

using QAOAHomology
import TensorCrossInterpolation as TCI
using QuanticsTCI
plotlyjs()

p = 3
γ1= (0.   , π/4)
γ = (-π/4 , π/4)

grid_specif = GridSpecifications(p, 20, γ1, γ)
pgrid = parameter_grid(grid_specif);

tolerance = 1e-6
@time tci, ranks, errors = tci_energy_grid(qaoa, pgrid, 10, false; tolerance=tolerance, verbosity=2);
TCI.cachedata(tci)


@time homology_ising = Homology0Data(p, 0., tci, pgrid, filtration_type=:Rips);

using Plots
plot(homology_ising.h0, homology_ising.merges, homology_ising.grid_extrema[2])

Γmat = rand(Uniform(-π/4, π/4), 2*3, 1000);
for i in 1:1000
    if Γmat[1, i] < 0
        Γmat[1, i] += π/4
    end
end

data_opt = warmOptimizeModulatedNewton(qaoa, Γmat);

histogram(data_opt[1], bins=100)
