harvardDictionary = Dict(
(1,2)=>0.38,
(1,3)=>0.15,
(1,5)=>0.52,
(2,3)=>0.76,
(2,4)=>0.19,
(3,7)=>0.16,
(6,7)=>0.9,
(5,6)=>0.66,
(6,9)=>0.43,
(4,8)=>0.82,
(5,8)=>0.7,
(4,9)=>0.76,
(7,12)=>0.56,
(8,10)=>0.16,
(9,11)=>0.89,
(10,11)=>0.69,
(10,13)=>0.36,
(12,14)=>0.37,
(12,13)=>0.47,
(11,14)=>0.76,
(13,14)=>0.14
);

function harvardGraph(T::Type{<:Real})
    graph = SimpleWeightedGraph(14)
    for k in keys(harvardDictionary)
        add_edge!(graph, k[1], k[2], harvardDictionary[k] |> T)
    end
    return graph
end

function labs_interactions(T::Type{<:Real}, n::Int)
    interaction_pairs = Dict{Vector{Int}, T}()
    for i ∈ 1:n-3
        for t ∈ 1:floor(Int, (n-i-1)/2)
            for k ∈ (t+1):(n-i-t)
                key = [i, i+t, i+k, i+k+t]
                if haskey(interaction_pairs, key)
                    interaction_pairs[key] += T(2)
                else
                    interaction_pairs[key] = T(2)
                end
            end
        end
    end
    for i ∈ 1:n-2
        for k ∈ 1:floor(Int, (n-i)/2)
            key = [i, i+2k]
            if haskey(interaction_pairs, key)
                interaction_pairs[key] += T(1)
            else
                interaction_pairs[key] = T(1)
            end
        end
    end
    return interaction_pairs
end