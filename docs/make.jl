using QAOALandscapes
using Documenter
using DocumenterMarkdown

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
        "QAOALandscapes" => "index.md",
        "api.md"
    ],
)

deploydocs(;
    repo="github.com/RaimelMedina/QAOALandscapes.jl",
    devbranch="main",
)

# makedocs(
#     sitename = "QAOALandscapes.jl",
#     #format = Documenter.HTML(
#     #    prettyurls = get(ENV, "CI", "false") == "true",
#     #    ansicolor = true
#     #),
#     format = Markdown(),
#     modules = [QAOALandscapes],
#     pages=[
#         "QAOALandscapes.jl" => "index.md",
#         "api.md"
#     ],
# )

# deploydocs(
#     repo="github.com/RaimelMedina/QAOALandscapes.jl",    
#     deps   = Deps.pip("mkdocs", "pygments", "python-markdown-math"),
#     make   = () -> run(`mkdocs build`),
#     devbranch="main",
#     target = "site"
# )