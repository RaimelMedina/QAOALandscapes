using QAOALandscapes
using JLD2
using ProgressMeter
using SimpleWeightedGraphs
using Graphs

N = 20
pmax = 40

folder_path = pwd()*"/examples/unweighted/"
files_with_extension = readdir(folder_path, join=true)

filtered_files = filter(f -> occursin("data_N_$(N)_pmax_40_", f), files_with_extension)


# filtered_files_unweighted = filter(f -> occursin("weighted_data_", f), filtered_pmax)
# #filtered_files_weighted = filter(f -> occursin("weighted_data_", f), files_with_extension)

# filtered_files = filter(f -> occursin("_N_"*string(N), f), filtered_files_unweighted)

@show "A total of $(length(filtered_files)) were found"

vec_of_data = Array{QAOAData, 1}(undef, length(filtered_files))

for (i,file) in enumerate(filtered_files)
    println("Working on file $(i) out of $(length(filtered_files))")
    vec_of_data[i] = jldopen(file)["data"]
end

vec_of_data[1]
jldsave("data_N_$(N)_pmax_$(pmax)_TS_1.jld2"; vec_of_data)

jldopen("data_N_20_pmax_40_TS_1.jld2")["vec_of_data"]