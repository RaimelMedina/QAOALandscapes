using QAOALandscapes
using Test
using LinearAlgebra
using Random
using Graphs

@testset "ClassicalProblem basics" begin
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    cp = ClassicalProblem(Float64, g)
    @test cp.n == 2
    @test cp.weightedQ == false

    ham_full = hamiltonian(cp, false)
    @test length(ham_full) == 4

    ham_sym = hamiltonian(cp)
    @test length(ham_sym) == 2

    weighted_terms = Dict([1, 2] => 2.0)
    cp_weighted = ClassicalProblem(weighted_terms, 2)
    @test cp_weighted.weightedQ == true
end

@testset "QAOA state construction" begin
    g = SimpleGraph(2)
    add_edge!(g, 1, 2)
    cp = ClassicalProblem(Float64, g)
    qaoa = QAOA(cp)

    Γ = zeros(Float64, 2)
    ψ = getQAOAState(qaoa, Γ)
    @test ψ ≈ qaoa.initial_state
    @test isapprox(norm(ψ), 1.0; atol=1e-10)

    ψ0 = copy(qaoa.initial_state)
    getQAOAState!(qaoa, Γ, ψ0)
    @test ψ0 ≈ qaoa.initial_state

    ψ1 = copy(qaoa.initial_state)
    ψ_custom = getQAOAState(qaoa, Γ, ψ1)
    @test ψ_custom ≈ ψ1
end

@testset "Initialization strategies" begin
    Random.seed!(1234)
    Γ = rand(4)

    interp = InterpInitialization()
    Γ_interp = interp(Γ)
    @test length(Γ_interp) == length(Γ) + 2

    Γ_ts = TSInitialization(Γ, 1, Val(:symmetric))
    @test length(Γ_ts) == length(Γ) + 2
    @test Γ_ts[1] == 0
    @test Γ_ts[2] == 0

    Γu = toFourierParams(Γ)
    @test length(Γu) == length(Γ)

    Γ_back = fromFourierParams(Γu)
    @test length(Γ_back) == length(Γ)

    Γ_fourier = fourierInitialization(Γ)
    @test length(Γ_fourier) == length(Γ) + 2
end

@testset "GPU backend selection" begin
    @test gpu_backend() in (:cpu, :metal, :cuda)
end
