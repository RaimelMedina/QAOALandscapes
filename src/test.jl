idSeed=0

function generateId()
    id = idSeed
    global idSeed = idSeed + 1
    #global idSeed ++
    return idSeed
end

println(generateId())