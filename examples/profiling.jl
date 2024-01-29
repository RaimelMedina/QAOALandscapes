using Revise
using QAOALandscapes

using BenchmarkTools
using ProfileView
using Graphs
using LinearAlgebra

N = 14

g = harvardGraph(Float64)
qaoa = QAOA(Float64, g)

Γ0 = rand(2);
toFundamentalRegion!(qaoa, Γtest)


@benchmark optimizeParameters(qaoa, Γtest)

@benchmark optimizeParameters(qaoa, Γtest)

qaoa(Γtest)
# @benchmark gradCostFunction(qaoa, Γtest)

@profview qaoa(Γtest)

ψ0 = plus_state(Float64, qaoa.N);
ψ1 = plus_state(Float64, qaoa.N);

@benchmark QAOALandscapes.applyExpHC!($qaoa.HC, 0.1, $ψ0)
@benchmark QAOALandscapes.applyExpHB!($ψ0, 0.1; parity_symmetry = true)

@benchmark QAOALandscapes.Hzz_ψ!($qaoa, $ψ0)

@benchmark QAOALandscapes.Hx_ψ!($qaoa, $ψ0)
@benchmark QAOALandscapes.Hx_ψ!($qaoa, $ψ0, $ψ1)



@benchmark QAOALandscapes.gradCostFunction($qaoa, $Γtest)
@benchmark QAOALandscapes.gradCostFunction_2($qaoa, $Γtest)

g1 = QAOALandscapes.gradCostFunction(qaoa, Γtest);
g2 = QAOALandscapes.gradCostFunction_2(qaoa, Γtest);

g1 ≈ g2

@benchmark QAOALandscapes.getQAOAState($qaoa, $Γtest)




@check_allocs QAOALandscapes.applyQAOALayer!($qaoa, $0.1, $100, $ψ0)

@benchmark QAOALandscapes.applyQAOALayerDerivative!($qaoa, 0.1, 100, $ψ0)

@code_warntype QAOALandscapes.gradCostFunction(qaoa, Γtest)