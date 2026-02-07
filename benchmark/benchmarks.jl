using QAOALandscapes
using BenchmarkTools
using Graphs

function setup_problem(n::Int, d::Int)
    g = random_regular_graph(n, d)
    prob = ClassicalProblem(Float64, g)
    qaoa = QAOA(prob)
    Γ = fill(0.1, 2)
    return qaoa, Γ
end

function run_benchmarks()
    qaoa, Γ = setup_problem(10, 3)

    println("Benchmark: hamiltonian")
    prob = qaoa.problem
    @btime hamiltonian($prob)

    println("Benchmark: getQAOAState")
    @btime getQAOAState($qaoa, $Γ)

    println("Benchmark: gradCostFunction")
    @btime gradCostFunction($qaoa, $Γ)
end

run_benchmarks()
