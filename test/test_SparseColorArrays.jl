ref_netcount(A, j, j′) = length(union(A.rowval[A.colptr[j]:A.colptr[j′] - 1]))

ref_netcount(A, Φ_domain, j, j′, k) = length(intersect(union(A[:, j:(j′ - 1)].rowval), Φ_domain.prm[Φ_domain.spl[k]:Φ_domain.spl[k + 1] - 1]))
ref_colnetcount(At, Φ_domain, j, j′, k) = length(intersect(union(At[:, Φ_domain.prm[Φ_domain.spl[k]:Φ_domain.spl[k + 1] - 1]].rowval), j:(j′ - 1)))

@testset "SparseColorArrays" begin

    for hint = ((), (ChainPartitioners.RandomHint(),), (ChainPartitioners.SparseHint(),), (ChainPartitioners.StepHint(),))
        for m = test_dims
            for n = test_dims
                A = dropzeros!(sprand(UInt, m, n, 0.5))
                net = netcount(hint..., A)
                @testset "hint=$hint, n = $n, m = $m" begin
                    for (j, j′) in test_points(10, 1:n + 1, 1:n + 1)
                        (j, j′) = minmax(j, j′)
                        @test net[j, j′] == ref_netcount(A, j, j′)

                        for _ = 1:10
                            _j, _j′= rand(filter(((_j, _j′),) -> destep(_j) < destep(_j′), collect(Iterators.product(Δ(j, 1:n + 1), Δ(j′, 1:n + 1)))))
                            (j, j′) = map(destep, (_j, _j′))
                            @test ChainPartitioners.Step(net)(_j, _j′) == ref_netcount(A, j, j′)
                        end
                    end
                end
            end
        end

        for m = test_dims, K = 1:8
            for n = test_dims
                Φ_map = MapPartition(K, rand(1:K, m))
                Φ_domain = convert(DomainPartition, Φ_map)
                n = m
                A = dropzeros!(sprand(UInt, m, n, 0.5))
                At = SparseMatrixCSC(A')
                if nnz(A) > 0
                    args, Ap = partwise(A, Φ_map)
                    rownet = partwisecost!(hint..., args..., netcount(hint..., Ap))
                    colnet = partwisecost!(hint..., args..., VertexCount())
                    @testset "hint=$hint, m = $m, n = $n, K = $K" begin
                        for (j, j′, k) in test_points(10, 1:n + 1, 1:n + 1, 1:K)
                            (j, j′) = minmax(j, j′)
                            k = rand(1:K)

                            @test rownet[j, j′, k] == ref_netcount(A, Φ_domain, j, j′, k)
                            @test colnet[j, j′, k] == ref_colnetcount(At, Φ_domain, j, j′, k)

                            for _ = 1:10
                                _j, _j′= rand(filter(((_j, _j′),) -> destep(_j) < destep(_j′), collect(Iterators.product(Δ(j, 1:n + 1), Δ(j′, 1:n + 1)))))
                                _k = rand(Δ(k, 1:K))
                                (j, j′, k) = map(destep, (_j, _j′, _k))
                                @test ChainPartitioners.Step(rownet)(_j, _j′, _k) == ref_netcount(A, Φ_domain, j, j′, k)
                                @test ChainPartitioners.Step(colnet)(_j, _j′, _k) == ref_colnetcount(At, Φ_domain, j, j′, k)
                            end
                        end
                    end
                end
            end
        end
    end
end