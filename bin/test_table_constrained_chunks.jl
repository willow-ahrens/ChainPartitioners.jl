using SparseArrays
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using UnicodePlots
using PrettyTables
using ChainPartitioners
using Cthulhu

for mtx in [
            "Boeing/ct20stif",
            "DIMACS10/chesapeake",
            #"Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))
    (m, n) = size(A)

    mdl = AffineNetCostModel{Int64}(0, 0, 0, 1)

    #println()
    #println()
    #println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    println(mtx)
    rows = []
    for (key, method) in [
        ("dynamic16", DynamicTotalChunker(ConstrainedCost(mdl, WidthCost{Int}(), fld(n, 16)))),
        ("quadrangle16", ConvexTotalChunker(ConstrainedCost(mdl, WidthCost{Int}(), fld(n, 16)))),
    ]
        setup_time = time(@benchmark pack_stripe($A, $method))
        Φ = pack_stripe(A, method)
        comm = total_value(A, Φ, mdl)
        @assert issorted(Φ.spl)
        @assert Φ.spl[1] == 1
        @assert Φ.spl[end] == n + 1
        @assert all((Φ.spl[2:end] .- Φ.spl[1:end-1]) .<= fld(n, 16))
        println(Φ.spl)

        push!(rows, [key setup_time comm])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "communication"])
end
