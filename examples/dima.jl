using QAOALandscapes
using LinearAlgebra

n = 24
p = 3

interaction_terms = Dict([i, i + 1] => 1.0 for i in 1:n-1)
for i in 1:n
    interaction_terms[[i]] = 1.0
end

prob = ClassicalProblem(interaction_terms, n)
qaoa = QAOA(prob)

Γ = fill(0.2, 2p)
ψ = getQAOAState(qaoa, Γ)

println("State norm: ", norm(ψ))
println("Energy: ", qaoa(Γ))