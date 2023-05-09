using QAOALandscapes
using JLD2
using LinearAlgebra
using ThreadsX
using Optim

qaoa = QAOA(harvardGraph; applySymmetries = true)
dir = pwd()
interpTest  = jldopen("/nfs/scistore14/serbyngrp/rmedinar/qaoa/QAOALandscapes/examples/Harvard_interp_p_22.jld2")["interpTest"]

pinit = 18
Γ0 = interpTest[pinit][2]
pmax = 30
toKeep = 4
dataGreedy = Vector[]
push!(dataGreedy, [Γ0])

for i ∈ 2:12
    push!(dataGreedy, rollDownTS(qaoa, dataGreedy[i-1], toKeep, threaded=true))
    println("Finished with p=$(pinit + i)")
end

jldsave("Harvard_greedy_p_18_30_keep_4.jld2"; dataGreedy)