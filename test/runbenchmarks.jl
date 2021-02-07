using BenchmarkTools
using ChainPartitioners
using SparseArrays
using JSON

include("matrices.jl")

function main(args)
    suite = BenchmarkGroup()

    suite["partition_stripe"] = BenchmarkGroup(["partition", "stripe"])
    suite["partition_plaid"] = BenchmarkGroup(["partition", "plaid"])
    suite["pack_stripe"] = BenchmarkGroup(["pack", "stripe"])

    work_model = (AffineWorkCostModel(0, 10, 1), "work_model")
    net_model = (AffineNetCostModel(0, 10, 1, 100), "net_model")
    sym_model = (AffineSymCostModel(0, 0, 1, 100, 90), "sym_model")
    comm_model = (AffineCommCostModel(0, 10, 1, 0, 100), "comm_model")
    local_model = (AffineLocalCostModel(0, 10, 1, 0, 100), "local_model")
    col_block_model = (ColumnBlockComponentCostModel{Int}(8, 3, (w) -> 1 + w), "col_block_model")
    block_model = (BlockComponentCostModel{Int}(8, 8, 1, 3, (1, identity), (1, identity)), "block_model")

    for mtx in [
        "Pajek/GD99_c",
        "LPnetlib/lp_blend",
    ]
        A = SparseMatrixCSC(matrices[mtx])
        (m, n) = size(A)
        ϵ = 0.01
        for (f, f_key) = [
            (DynamicTotalChunker(net_model[1], 8), "DynamicTotalChunker(net_model, 8)"),
            (DynamicTotalChunker(col_block_model[1], 8), "DynamicTotalChunker(col_block_model, 8)"),
            (OverlapChunker(0.9, 8), "OverlapChunker(0.9, 8)"),
            (StrictChunker(8), "StrictChunker(8)"),
        ]
            suite["pack_stripe"]["pack_stripe($(mtx), $f_key)"] = @benchmarkable pack_stripe($A, $f)
        end
        for (Π, Π_key) in [
            (pack_stripe(A', EquiChunker(4)), "Π_spl"),
        ]
            for (f, f_key) = [
                (DynamicTotalChunker(block_model[1], 8), "DynamicTotalChunker(block_model, 8)"),
            ]
                suite["pack_stripe"]["pack_stripe($(mtx), $f_key, $Π_key)"] = @benchmarkable pack_stripe($A, $f, $Π)
            end
        end
        for K = [2, ceil(Int, n^(3/4))]
            mdls = [
                work_model,
                net_model,
            ]
            if issymmetric(pattern(A))
                append!(mdls, [
                    sym_model
                ])
            end
            for (f, f_key) = [
                (EquiSplitter(), "EquiSplitter()"),
                ((BisectIndexBottleneckSplitter(mdl), "BisectIndexBottleneckSplitter($mdl_key)") for (mdl, mdl_key) in mdls)...,
                ((BisectCostBottleneckSplitter(mdl, ϵ), "BisectCostBottleneckSplitter($mdl_key)") for (mdl, mdl_key) in mdls)...,
                ((LazyBisectCostBottleneckSplitter(mdl, ϵ), "LazyBisectCostBottleneckSplitter($mdl_key)") for (mdl, mdl_key) in mdls[2:end])...,
                (DynamicTotalSplitter(net_model[1]), "DynamicTotalSplitter(net_model)"),
            ]
                suite["partition_stripe"]["partition_stripe($(mtx), $K, $f_key)"] = @benchmarkable partition_stripe($A, $K, $f)
            end
            for (Π, Π_key, mdls) in [
                (partition_stripe(A', K, EquiSplitter()), "Π_spl", [local_model, comm_model]),
                (MapPartition(K, mod1.(1:m, K)), "Π_map", [comm_model,]),
            ]
                for (f, f_key) = [
                    ((BisectIndexBottleneckSplitter(mdl), "BisectIndexBottleneckSplitter($mdl_key)") for (mdl, mdl_key) in mdls)...,
                    ((BisectCostBottleneckSplitter(mdl, ϵ), "BisectCostBottleneckSplitter($mdl_key)") for (mdl, mdl_key) in mdls)...,
                    (LazyBisectCostBottleneckSplitter(comm_model[1], ϵ), "LazyBisectCostBottleneckSplitter(comm_model)"),
                    (GreedyBottleneckPartitioner(local_model[1]), "GreedyBottleneckPartitioner(local_model)"),
                    (MagneticPartitioner(), "MagneticPartitioner"),
                ]
                    suite["partition_stripe"]["partition_stripe($(mtx), $K, $f_key, $Π_key)"] = @benchmarkable partition_stripe($A, $K, $f, $Π)
                end
            end
        end
    end

    if length(args) == 1 && args[1] == "update"
        tune!(suite, verbose=true, seconds=60)
        BenchmarkTools.save(joinpath(@__DIR__, "params.json"), BenchmarkTools.params(suite))
        results = run(suite, verbose=true, seconds=60)
        BenchmarkTools.save(joinpath(@__DIR__, "benchmarks.json"), results)
        return
    elseif length(args) == 0
        params = BenchmarkTools.load(joinpath(@__DIR__, "params.json"))[1]
        BenchmarkTools.loadparams!(suite, params)
        results = run(suite, seconds=60, verbose=true)
        reference = BenchmarkTools.load(joinpath(@__DIR__, "benchmarks.json"))[1]
        judgements = judge(minimum(results), minimum(reference); time_tolerance = 0.05)
        show(IOContext(stdout, :compact=>false, :limit=>false), regressions(judgements))
        show(IOContext(stdout, :compact=>false, :limit=>false), improvements(judgements))
        println()
        return
    elseif length(args) == 1
        key = args[1]
        params = BenchmarkTools.load(joinpath(@__DIR__, "params.json"))[1]
        BenchmarkTools.loadparams!(suite, params)
        results = run(suite[key], seconds=60, verbose=true)
        reference = BenchmarkTools.load(joinpath(@__DIR__, "benchmarks.json"))[1][key]
        judgements = judge(minimum(results), minimum(reference); time_tolerance = 0.05)
        show(IOContext(stdout, :compact=>false, :limit=>false), regressions(judgements))
        #show(IOContext(stdout, :compact=>false, :limit=>false), improvements(judgements))
        println()
        return
    else
        throw(ArgumentError("unrecognized command line arguments"))
    end
end

main(ARGS)
