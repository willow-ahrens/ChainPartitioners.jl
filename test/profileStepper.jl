using Profile
using Cthulhu
using ChainPartitioners
using ChainPartitioners: StepHint, Step, Next, Same, Prev
using SparseArrays
using MatrixDepot

function stepstop(f::F, n) where {F}
    @inbounds begin
        f(1, 1, 1)
        for j′ = 2:n
            Step(f)(Same(1), Next(j′), Same(1))
            f(1, j′, 1)
            for j = 2:j′
                Step(f)(Next(j), Same(j′), Same(1))
            end
        end
    end
end

function main()
    work_model = (AffineWorkModel(0, 10, 1), "work_model")
    net_model = (AffineConnectivityModel(0, 10, 1, 100), "net_model")
    sym_model = (AffineMonotonizedSymmetricConnectivityModel(0, 0, 1, 100, 90), "sym_model")
    comm_model = (AffinePrimaryConnectivityModel(0, 10, 1, 0, 100), "comm_model")
    local_model = (AffineSecondaryConnectivityModel(0, 10, 1, 0, 100), "local_model")
    col_block_model = (ColumnBlockComponentCostModel{Int}(3, (w) -> 1 + w), "col_block_model")
    block_model = (BlockComponentCostModel{Int}(1, 3, (1, identity), (1, identity)), "block_model")

    A = SparseMatrixCSC(mdopen("Boeing/ct20stif").A)
    (m, n) = size(A)
    f = oracle_stripe(StepHint(), net_model[1], A, partition_stripe(A', 32, EquiSplitter()))

    stepstop(f, n)
    @profile for i = 1:10 stepstop(f, n) end
    Profile.print()
end

main()