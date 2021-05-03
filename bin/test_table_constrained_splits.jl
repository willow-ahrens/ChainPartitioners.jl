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
            "Boeing/ct20stif",
            #"Schmid/thermal1",
            #"Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))
    (m, n) = size(A)

    y = zeros(m)
    x = rand(n)
    ref_time = @belapsed(mul!($y, $A, $x))

    println(mtx)
    rows = []
    for K = [4, 8, 16]
        mdl = ConstrainedCost(AffineConnectivityModel{Int64}(0, 0, 0, 1), VertexCount(), ceil(Int, n/K * 1.5))

        (j_lo, j′_hi) = ChainPartitioners.column_constraints(A, K, mdl.w, mdl.w_max)
        show(histogram(j′_hi .- j_lo, nbins = 15, closed = :left))
        println()

        #println()
        #println()
        #println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
        for (key, method) in [
            ("dynamic", DynamicTotalSplitter(mdl)),
            ("dynamic′", DynamicTotalChunker(mdl)),
            ("quadrangle", ConvexTotalSplitter(mdl)),

        ]
            setup_time = @belapsed(partition_stripe($A, $K, $method))/ref_time
            Φ = nothing
            #@profile begin
                Φ = partition_stripe(A, K, method)
            #end
            #Profile.print()
            comm = total_value(A, Φ, mdl)
            @assert issorted(Φ.spl)
            @assert Φ.spl[1] == 1
            @assert Φ.spl[end] == n + 1

            push!(rows, [key K setup_time comm])
        end
    end
    pretty_table(vcat(rows...), ["method", "K", "setuptime", "communication"])
end
