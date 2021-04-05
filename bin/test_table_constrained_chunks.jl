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
            #"HB/can_292",
            #"HB/662_bus",
            #"HB/fs_680_2",
            #"HB/can_292",
            #"HB/plat362",
            #"LPnetlib/lp_bandm",
            #"LPnetlib/lp_etamacro",
            #"Pajek/Erdos991",
            "Boeing/ct20stif",
            #"DIMACS10/chesapeake",
            "Schmid/thermal1",
            "Rothberg/3dtube",
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
        ("dynamic16", DynamicTotalChunker(ConstrainedCost(mdl, WidthCost(), fld(n, 4)))),
        ("quadrangle16", ConvexTotalChunker(ConstrainedCost(mdl, WidthCost(), fld(n, 4)))),
    ]
        setup_time = @belapsed(pack_stripe($A, $method))/ref_time
        Φ = pack_stripe(A, method)
        #@profile pack_stripe(A, method)
        #Profile.print()
        comm = total_value(A, Φ, mdl)
        @assert issorted(Φ.spl)
        @assert Φ.spl[1] == 1
        @assert Φ.spl[end] == n + 1

        push!(rows, [key setup_time comm])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "communication"])
end
