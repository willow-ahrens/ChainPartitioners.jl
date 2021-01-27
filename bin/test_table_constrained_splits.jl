using SparseArrays
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using UnicodePlots
using PrettyTables
using ChainPartitioners
using Cthulhu
using Profile

for mtx in [
            "DIMACS10/chesapeake",
            #"HB/can_292",
            "Boeing/ct20stif",
            "Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))
    (m, n) = size(A)

    y = zeros(m)
    x = rand(n)
    ref_time = @belapsed(mul!($y, $A, $x))

    mdl = ConstrainedCost(AffineNetCostModel{Int64}(0, 0, 0, 1), WidthCost{Int}(), cld(n, 8))

    #println()
    #println()
    #println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    println(mtx)
    rows = []
    for (key, method) in [
        ("dynamic16", DynamicTotalSplitter(mdl)),
        ("dynamic′16", DynamicTotalChunker(mdl)),
        ("quadrangle16", ConvexTotalSplitter(mdl)),

    ]
        setup_time = @belapsed(partition_stripe($A, 16, $method))/ref_time
        Φ = partition_stripe(A, 16, method)
        #@descend pack_stripe(A, method)
        comm = total_value(A, Φ, mdl)
        @assert issorted(Φ.spl)
        @assert Φ.spl[1] == 1
        @assert Φ.spl[end] == n + 1

        push!(rows, [key setup_time comm])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "communication"])
end
