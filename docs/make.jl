using ChainPartitioners
using Documenter

makedocs(;
    modules=[ChainPartitioners],
    authors="Peter Ahrens <ptrahrens@gmail.com> and contributors",
    repo="https://github.com/peterahrens/ChainPartitioners.jl/blob/{commit}{path}#L{line}",
    sitename="ChainPartitioners.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://peterahrens.github.io/ChainPartitioners.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/peterahrens/ChainPartitioners.jl",
)
