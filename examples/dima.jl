using QAOALandscapes
using LinearAlgebra
n = 24
num_cicles = 10
interaction_terms = Dict([i, i+1] => 1.0 for i in 1:n-1)
for i in 1:n
    interaction_terms[[i]] = 1.0
end

prob = ClassicalProblem(interaction_terms, n)

qaoa = QAOA(prob)

# qaoa.HC = ∑ Z_i Z_j + ...
# qaoa.mixer = HB = ∑ᵢ σˣᵢ

Γ = ones(2*num_cicles) ./ 2

# Γ = (γ₁, β₁, ..., γₚ, βₚ)
# ψ = ∏ᵢ UB(βᵢ) UC(γᵢ) |+⟩
# UB(β) = exp(-i β HB)
# UC(γ) = exp(-i γ HC)
ψ = getQAOAState(qaoa, Γ)

norm(ψ)

using Yao

ψY = ArrayReg(ψ)
von_neumann_entropy(ψY, (1,2))