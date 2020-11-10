using BenchmarkTools
using ChainPartitioners
using SparseArrays
using JSON

include("matrices.jl")

function main(args)
    suite = BenchmarkGroup()

    suite["partition_stripe"] = BenchmarkGroup(["partition", "stripe"])
    suite["partition_plaid"] = BenchmarkGroup(["partition", "plaid"])

    work_model = (AffineWorkCostModel(0, 10, 1), "work_model")
    net_model = (AffineNetCostModel(0, 10, 1, 100), "net_model")
    sym_model = (AffineSymCostModel(0, 0, 1, 100, 90), "sym_model")
    comm_model = (AffineCommCostModel(0, 10, 1, 0, 100), "comm_model")
    local_model = (AffineLocalCostModel(0, 10, 1, 0, 100), "local_model")

    for mtx in [
        "Pajek/GD99_c",
        "LPnetlib/lp_blend",
    ]
        A = SparseMatrixCSC(matrices[mtx])
        (m, n) = size(A)
        ϵ = 0.01
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
                (EquiPartitioner(), "EquiPartitioner()"),
                ((NicolPartitioner(mdl), "NicolPartitioner($mdl_key)") for (mdl, mdl_key) in mdls)...,
                ((BisectPartitioner(mdl, ϵ), "BisectPartitioner($mdl_key)") for (mdl, mdl_key) in mdls)...,
                ((LazyBisectPartitioner(mdl, ϵ), "LazyBisectPartitioner($mdl_key)") for (mdl, mdl_key) in mdls[2:end])...,
            ]
                suite["partition_stripe"]["partition_stripe($(mtx), $K, $f_key)"] = @benchmarkable partition_stripe($A, $K, $f)
            end
            for (Π, Π_key, mdls) in [
                (partition_stripe(A', K, EquiPartitioner()), "Π_spl", [local_model, comm_model]),
                (MapPartition(K, mod1.(1:m, K)), "Π_map", [comm_model,]),
            ]
                for (f, f_key) = [
                    ((NicolPartitioner(mdl), "NicolPartitioner($mdl_key)") for (mdl, mdl_key) in mdls)...,
                    ((BisectPartitioner(mdl, ϵ), "BisectPartitioner($mdl_key)") for (mdl, mdl_key) in mdls)...,
                    (LazyBisectPartitioner(comm_model[1], ϵ), "LazyBisectPartitioner(comm_model)"),
                    (GreedyLocalCostPartitioner(local_model[1]), "GreedyLocalCostPartitioner(local_model)"),
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
    elseif length(args) == 0
        params = BenchmarkTools.load(joinpath(@__DIR__, "params.json"))[1]
        BenchmarkTools.loadparams!(suite, params)
        results = run(suite, seconds=60, verbose=true)
        reference = BenchmarkTools.load(joinpath(@__DIR__, "benchmarks.json"))[1]
        judgements = judge(minimum(results), minimum(reference); time_tolerance = 0.05)
        show(IOContext(stdout, :compact=>false, :limit=>false), regressions(judgements))
        println()
    else
        throw(ArgumentError("unrecognized command line arguments"))
    end
end

main(ARGS)