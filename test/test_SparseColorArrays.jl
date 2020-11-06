@testset "SparseColorArrays" begin
    trials = 100

    for m = 1:40, H = 1:4
        for n = 1:40
            A = dropzeros!(sprand(UInt, m, n, 0.5))
            net = rownetcount(A)
            @testset "H = $H, m = $m" begin
                for _ = 1:trials
                    j = rand(1:n + 1)
                    j′ = rand(j:n + 1)
                    if !((@test net[j, j′] == length(union(A.rowval[A.colptr[j]:A.colptr[j′] - 1]))) isa Test.Pass)
                        @info "debugging stuff:" A, j, j′
                        break
                    end
                end
            end
        end
    end

    for m = 1:40, K = 1:8
        for n = 1:40
            Φ_map = MapPartition(K, rand(1:K, m))
            Φ_domain = convert(DomainPartition, Φ_map)
            n = m
            A = dropzeros!(sprand(UInt, m, n, 0.5))
            At = SparseMatrixCSC(A')
            if nnz(A) > 0
                rownet = localrownetcount(A, K, Φ_map)
                colnet = localcolnetcount(A, K, Φ_map)
                @testset "m = $m, K = $K" begin
                    for _ = 1:trials
                        j = rand(1:n + 1)
                        j′ = rand(j:n + 1)
                        k = rand(1:K)
                        if !((@test rownet[j, j′, k] == length(intersect(union(A[:, j:(j′ - 1)].rowval), Φ_domain.prm[Φ_domain.spl[k]:Φ_domain.spl[k + 1] - 1]))) isa Test.Pass)
                            @info "debugging stuff:" A, j, j′
                            break
                        end
                        if !((@test colnet[j, j′, k] == length(intersect(union(At[:, Φ_domain.prm[Φ_domain.spl[k]:Φ_domain.spl[k + 1] - 1]].rowval), j:(j′ - 1)))) isa Test.Pass)
                            display(spy(A))
                            println()
                            @info "debugging stuff:" j j′ Φ k
                            exit()
                            break
                        end
                    end
                end
            end
        end
    end
end