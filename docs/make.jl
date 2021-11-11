using WassersteinDictionaries
using Documenter

DocMeta.setdocmeta!(WassersteinDictionaries, :DocTestSetup, :(using WassersteinDictionaries); recursive=true)

makedocs(;
    modules=[WassersteinDictionaries],
    authors="Tobias M. Blickhan, Michael Kraus",
    repo="https://github.com/JuliaRCM/WassersteinDictionaries.jl/blob/{commit}{path}#{line}",
    sitename="WassersteinDictionaries.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaRCM.github.io/WassersteinDictionaries.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaRCM/WassersteinDictionaries.jl",
    devbranch="main",
)
