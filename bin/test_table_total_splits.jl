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
            #"DIMACS10/chesapeake",
            #"HB/can_292",
            "Pajek/Erdos991",
            "PARSEC/SiNa",
            #"Boeing/ct20stif",
            #"Schmid/thermal1",
            #"Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))
    (m, n) = size(A)

    y = zeros(m)
    x = rand(n)
    ref_time = @belapsed(mul!($y, $A, $x))

    mdl = AffineNetCostModel{Int64}(0, 0, 0, 1)

    #println()
    #println()
    #println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    println(mtx)
    rows = []
    for (key, method) in [
        ("dynamic16", DynamicTotalSplitter(mdl)),
        ("quadrangle16", ConvexTotalChunker(mdl)),
    ]
        setup_time = @belapsed(partition_stripe($A, 4, $method))/ref_time
        Φ = partition_stripe(A, 4, method)
        #@descend pack_stripe(A, method)
        comm = total_value(A, Φ, mdl)
        @assert issorted(Φ.spl)
        @assert Φ.spl[1] == 1
        @assert Φ.spl[end] == n + 1

        push!(rows, [key setup_time comm])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "communication"])
end
