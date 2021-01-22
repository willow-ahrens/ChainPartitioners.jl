@testset "SparsePrefixMatrices" begin
    trials = 100

    @testset "generic" begin
        for m = 1:40, kwargs = ((H = 1,), (H = 2,), (H = 3,), (H = 4,), (b = 1,), (b = 2,), (b = 3,), (b = 4,))
            for n = 1:40
                A = dropzeros!(sprand(Int, m, n, 0.5))
                C = areacount(ChainPartitioners.SparseHint(), A; kwargs...)
                @test typeof(C) <: SparseCountedArea
                S = areasum(A; kwargs...)
                @test typeof(S) <: SparseSummedArea
                @testset "kwargs = $kwargs, m = $m" begin
                    for _ = 1:trials
                        i = rand(0:m)
                        j = rand(0:n)
                        @test C[i + 1, j + 1] == sum(A[1:i, 1:j] .!= 0)
                        @test S[i + 1, j + 1] == sum(A[1:i, 1:j])
                    end
                end

                @test C[1, 1] == 0
                @test C[1, n + 1] == 0
                @test C[m + 1, 1] == 0
                @test C[m + 1, n + 1] == nnz(A)

                @test S[1, 1] == 0
                @test S[1, n + 1] == 0
                @test S[m + 1, 1] == 0
                @test S[m + 1, n + 1] == sum(A)

                N = m
                idx = randperm(N)
                B = spzeros(UInt, N, N)
                for j = 1:N
                    B[idx[j], j] = (rand(UInt) << 1) + 1
                end
                RC = rookcount!(ChainPartitioners.SparseHint(), N, copy(idx); kwargs...)
                @test typeof(RC) <: SparseCountedRooks
                RS = rooksum!(N, copy(idx), B.nzval; kwargs...)
                @test typeof(RS) <: SparseSummedRooks
                @testset "kwargs = $kwargs, N = $N" begin
                    for _ = 1:trials
                        i = rand(0:N)
                        j = rand(0:N)
                        @test RC[i + 1, j + 1] == sum(B[1:i, 1:j] .!= 0)
                        @test RS[i + 1, j + 1] == sum(B[1:i, 1:j])
                    end
                end
                @test RC[1, 1] == 0
                @test RC[1, N + 1] == 0
                @test RC[N + 1, 1] == 0
                @test RC[N + 1, N + 1] == sum(B .!= 0)

                @test RS[1, 1] == 0
                @test RS[1, N + 1] == 0
                @test RS[N + 1, 1] == 0
                @test RS[N + 1, N + 1] == sum(B)
            end
        end
    end

    @testset "binary" begin
        for m = 1:40
            for n = 1:40
                A = dropzeros!(sprand(Int, m, n, 0.5))
                C = areacount(A, b=1)
                @test typeof(C) <: SparseBinaryCountedArea
                @testset "m = $m" begin
                    for _ = 1:trials
                        i = rand(0:m)
                        j = rand(0:n)
                        @test C[i + 1, j + 1] == sum(A[1:i, 1:j] .!= 0)
                    end
                end

                @test C[1, 1] == 0
                @test C[1, n + 1] == 0
                @test C[m + 1, 1] == 0
                @test C[m + 1, n + 1] == nnz(A)

                N = m
                idx = randperm(N)
                B = spzeros(UInt, N, N)
                for j = 1:N
                    B[idx[j], j] = (rand(UInt) << 1) + 1
                end
                RC = rookcount!(N, copy(idx); b=1)
                @test typeof(RC) <: SparseBinaryCountedRooks
                @testset "N = $N" begin
                    for _ = 1:trials
                        i = rand(0:N)
                        j = rand(0:N)
                        @test RC[i + 1, j + 1] == sum(B[1:i, 1:j] .!= 0)
                    end
                end
                @test RC[1, 1] == 0
                @test RC[1, N + 1] == 0
                @test RC[N + 1, 1] == 0
                @test RC[N + 1, N + 1] == sum(B .!= 0)
            end
        end
    end
end