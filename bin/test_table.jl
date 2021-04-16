using SparseArrays
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using UnicodePlots
using PrettyTables
using ChainPartitioners

for mtx in [
            "Boeing/ct20stif",
            "DIMACS10/chesapeake",
            #"Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))

    mdl = BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (4, identity))

    println()
    println()
    println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    rows = []
    for (key, method) in [
        ("original", nothing),
        ("1D", AlternatingPacker(EquiChunker(1), DynamicTotalChunker(mdl, 8))),
        ("strict", AlternatingPacker(StrictChunker(8), StrictChunker(8))),
        ("overlap", AlternatingPacker(OverlapChunker(0.9, 8), OverlapChunker(0.9, 8))),
        ("dynamic", AlternatingPacker(DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 4), 8), DynamicTotalChunker(mdl, 8))),
        ("dynamic2", AlternatingPacker(
            DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 4), 8),
            DynamicTotalChunker(mdl, 8),
            DynamicTotalChunker(mdl, 8),
            DynamicTotalChunker(mdl, 8),
            DynamicTotalChunker(mdl, 8),
        )),
        ("dynamic3", AlternatingPacker(
            DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 1), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (2, identity)), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (3, identity)), 8),
            DynamicTotalChunker(mdl, 8),
            DynamicTotalChunker(mdl, 8)
        )),
        ("dynamic4", AlternatingPacker(
            DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 7), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (6, identity)), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (5, identity)), 8),
            DynamicTotalChunker(mdl, 8),
            DynamicTotalChunker(mdl, 8)
        )),
        #=
        ("strict_sym", SymmetricPacker(StrictChunker(8))),
        ("overlap_sym", SymmetricPacker(OverlapChunker(0.9, 8))),
        ("dynamic_sym", SymmetricPacker(DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 4), 8), DynamicTotalChunker(mdl, 8))),
        ("dynamic2_sym", SymmetricPacker(
            DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 4), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (4, identity)), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (4, identity)), 8),
            DynamicTotalChunker(mdl, 8)
        )),
        ("dynamic3_sym", SymmetricPacker(
            DynamicTotalChunker(AffineFillConnectivityModel(0, 0, 1, 1), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (2, identity)), 8),
            DynamicTotalChunker(BlockComponentCostModel{Int64}((8, 8), 0, 0, (1, identity), (3, identity)), 8),
            DynamicTotalChunker(mdl, 8)
        )),
        =#
    ]
        if method === nothing
            setup_time = 0
            memory = 5 * nnz(A)
        else
            setup_time = time(@benchmark (Π, Φ) = pack_plaid($A, $method))
            (Π, Φ) = pack_plaid(A, method)
            memory = total_value(A, Π, Φ, mdl)
        end

        push!(rows, [key setup_time memory])
    end
    pretty_table(vcat(rows...), ["method", "setuptime", "memory"])
end
