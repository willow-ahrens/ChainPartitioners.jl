using SparseArrays
using LinearAlgebra
using MatrixDepot
using BenchmarkTools
using UnicodePlots
using PrettyTables
using ChainPartitioners

for mtx in [
            "DIMACS10/chesapeake",
            "Boeing/ct20stif",
            #"Schmid/thermal1",
            "Rothberg/3dtube",
           ]
    A = permutedims(1.0 * sparse(mdopen(mtx).A))

    work_model = AffineWorkCostModel(0, 10, 1)
    net_model = AffineConnectivityModel(0, 10, 1, 100)
    sym_model = AffineSymmetricConnectivityModel(0, 0, 1, 100, 90)
    comm_model = AffinePrimaryConnectivityModel(0, 10, 1, 0, 100)
    loc_model = AffineSecondaryConnectivityModel(0, 10, 1, 0, 100)
    eps=0.1

    println()
    println()
    println(spy(A, maxwidth=50, maxheight=50, title="$mtx"))
    rows = []
    for (method, key, rng) in [
        (DisjointPartitioner(EquiSplitter(), EquiSplitter()), "split_equal__split_equal", false),
        (DisjointPartitioner(BisectIndexBottleneckSplitter(work_model), EquiSplitter()), "split_work__split_equal", false),
        (AlternatingPartitioner(BisectIndexBottleneckSplitter(work_model), MagneticPartitioner()), "split_work__map_local", true),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), FlipBisectIndexBottleneckSplitter(loc_model)), "split_nets__split_comm", false),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), FlipBisectIndexBottleneckSplitter(loc_model), BisectIndexBottleneckSplitter(comm_model), FlipBisectIndexBottleneckSplitter(loc_model)), "split_nets__split_comm__split_comm__split_comm", false),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), MagneticPartitioner()), "split_nets__map_local", true),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), MagneticPartitioner(), BisectIndexBottleneckSplitter(comm_model)), "split_nets__map_local__split_comm", true),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), GreedyBottleneckPartitioner(loc_model)), "split_nets__map_greedy", true),
        (AlternatingNetPartitioner(BisectIndexBottleneckSplitter(net_model), GreedyBottleneckPartitioner(loc_model), BisectIndexBottleneckSplitter(comm_model)), "split_nets__map_greedy__split_comm", true),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps)), "split_nets__split_comm__approx", false),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps), BisectCostBottleneckSplitter(comm_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps)), "split_nets__split_comm__split_comm__split_comm__approx", false),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), MagneticPartitioner()), "split_nets__map_local__approx", true),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), MagneticPartitioner(), BisectCostBottleneckSplitter(comm_model, eps)), "split_nets__map_local__split_comm__approx", true),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), GreedyBottleneckPartitioner(loc_model)), "split_nets__map_greedy__approx", true),
        (AlternatingNetPartitioner(BisectCostBottleneckSplitter(net_model, eps), GreedyBottleneckPartitioner(loc_model), BisectCostBottleneckSplitter(comm_model, eps)), "split_nets__map_greedy__split_comm__approx", true),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps)), "split_nets__split_comm__lazy__approx", false),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps), LazyBisectCostBottleneckSplitter(comm_model, eps), FlipBisectCostBottleneckSplitter(loc_model, eps)), "split_nets__split_comm__split_comm__split_comm__lazy__approx", false),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), MagneticPartitioner()), "split_nets__map_local__lazy__approx", true),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), MagneticPartitioner(), LazyBisectCostBottleneckSplitter(comm_model, eps)), "split_nets__map_local__split_comm__lazy__approx", true),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), GreedyBottleneckPartitioner(loc_model)), "split_nets__map_greedy__lazy__approx", true),
        (AlternatingPartitioner(LazyBisectCostBottleneckSplitter(net_model, eps), GreedyBottleneckPartitioner(loc_model), LazyBisectCostBottleneckSplitter(comm_model, eps)), "split_nets__map_greedy__split_comm__lazy__approx", true),
    ]
        for K in [32, 512]
            setup_time = time(@benchmark (Π, Φ) = partition_plaid($A, $K, $method))
            (Π, Φ) = partition_plaid(A, K, method)
            push!(rows, [K key setup_time])
            println(method)
        end
    end
    pretty_table(vcat(rows...), ["K", "method", "setuptime"])
end
