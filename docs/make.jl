using OptimalTransportTools
using Documenter

DocMeta.setdocmeta!(OptimalTransportTools, :DocTestSetup, :(using OptimalTransportTools); recursive=true)

makedocs(;
    modules=[OptimalTransportTools],
    authors="Tobias M. Blickhan, Michael Kraus",
    repo="https://github.com/JuliaRCM/OptimalTransportTools.jl/blob/{commit}{path}#{line}",
    sitename="OptimalTransportTools.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaRCM.github.io/OptimalTransportTools.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaRCM/OptimalTransportTools.jl",
    devbranch="main",
)
