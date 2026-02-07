using QAOALandscapes
using Graphs
using Random

Random.seed!(1234)

n = 10
d = 3
pmax = 3
g = random_regular_graph(n, d)

prob = ClassicalProblem(Float64, g)
qaoa = QAOA(prob)

Γ0, E0 = getInitialParameter(qaoa)
println("Initial energy (p=1): ", E0)

interp = InterpInitialization()
energies, params = optimizeWithStrategy(qaoa, Γ0, pmax, interp, OptimizationOptimJL.BFGS())
println("Energies by depth: ", energies)
