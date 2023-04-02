using QAOALandscapes
using Documenter

DocMeta.setdocmeta!(QAOALandscapes, :DocTestSetup, :(using QAOALandscapes); recursive=true)

makedocs(;
    modules=[QAOALandscapes],
    authors="Raimel A. Medina",
    repo="https://github.com/RaimelMedina/QAOALandscapes.jl/blob/{commit}{path}#{line}",
    sitename="QAOALandscapes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RaimelMedina.github.io/QAOALandscapes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RaimelMedina/QAOALandscapes.jl",
    devbranch="main",
)
