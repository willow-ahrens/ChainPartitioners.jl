@testset "Costs" begin
    trials = 100

    for m = 1:100, K = 1:4
        n = m
        A = dropzeros!(sprand(Int, m, n, 0.125))
        At = SparseMatrixCSC(A')
        Π = MapPartition(K, rand(1:K, m))
        Φ = SplitPartition(K, [1, sort(rand(1:(n + 1), K - 1))..., n + 1])

        models = [
            AffineWorkCostModel(0, 0, 1),
            AffineWorkCostModel(0, 1, 0),
            AffineWorkCostModel(0, 1, 1),
            AffineNetCostModel(0, 0, 0, 1),
            AffineNetCostModel(0, 0, 1, 0),
            AffineNetCostModel(0, 0, 1, 1),
            AffineNetCostModel(0, 1, 0, 0),
            AffineNetCostModel(0, 1, 0, 1),
            AffineNetCostModel(0, 1, 1, 0),
            AffineNetCostModel(0, 1, 1, 1),
            AffineSymCostModel(10, 10, 10, 10, 0),
            AffineSymCostModel(10, 10, 10, 100, 8),
        ]
        for mdl in models
            ocl = oracle_stripe(mdl, A, Φ)
            @test bottleneck_value(A, Π, Φ, mdl) == bottleneck_value(A, Π, Φ, ocl)
            @test bound_stripe(A, K, Π, mdl) == bound_stripe(A, K, Π, ocl)
            c_lo, c_hi = bound_stripe(A, K, Π, mdl)
            @test 0 <= c_lo <= bottleneck_value(A, Π, Φ, mdl) <= c_hi
        end

        Π = SplitPartition(K, [1, sort(rand(1:(m + 1), K - 1))..., m + 1])
        models = [
            (AffineCommCostModel(0, 0, 0, 0, 1), AffineLocalCostModel(0, 0, 0, 0, 1)),
            (AffineCommCostModel(0, 0, 0, 1, 1), AffineLocalCostModel(0, 0, 0, 1, 1)),
            (AffineCommCostModel(1, 1, 1, 1, 1), AffineLocalCostModel(1, 1, 1, 1, 1)),
        ]
        adj_A = permutedims(A)
        for (comm_mdl, local_mdl) in models
            comm_ocl = oracle_stripe(comm_mdl, A, Π)
            local_ocl = oracle_stripe(local_mdl, adj_A, Φ)
            @test bottleneck_value(A, Π, Φ, comm_mdl) == bottleneck_value(A, Π, Φ, comm_ocl)
            @test bottleneck_value(adj_A, Φ, Π, local_mdl) == bottleneck_value(adj_A, Φ, Π, local_ocl)
            @test bound_stripe(A, K, Π, comm_mdl) == bound_stripe(A, K, Π, comm_ocl)
            @test bound_stripe(adj_A, K, Φ, local_mdl) == bound_stripe(adj_A, K, Φ, local_ocl)
            c_lo, c_hi = bound_stripe(A, K, Π, comm_mdl)
            @test 0 <= c_lo <= bottleneck_value(A, Π, Φ, comm_mdl) <= c_hi
            c_lo, c_hi = bound_stripe(adj_A, K, Φ, local_mdl)
            @test 0 <= c_lo <= bottleneck_value(adj_A, Φ, Π, local_mdl) <= c_hi
            @test bottleneck_value(adj_A, Φ, Π, local_mdl) == bottleneck_value(A, Π, Φ, comm_mdl)
            @test bottleneck_value(adj_A, Φ, Π, local_ocl) == bottleneck_value(A, Π, Φ, comm_ocl)
        end
    end

    #TODO add wrapper cost functions to test that oracle versions of partitioners, bounds, and cost evaluators agree with direct versions.

    #=
    for m = 1:100, h = 1:4
        n = m
        k = 1000
        A = dropzeros!(sprand(UInt, m, n, 0.5))
        c = CommunicationCostFamily((l, w, d) -> d)(A)
        @testset "h = $h, m = $m" begin
            for l = 1:k
                j = rand(1:n)
                j′ = rand(j:n) + 1
                if !((@test c(j, j′) == length(union(A.rowval[A.colptr[j]:A.colptr[j′] - 1]))) isa Test.Pass)
                    @info "debugging stuff:" A, j, j′
                    break
                end
            end
        end
    end
    =#
end