@testset "SparsePrefixMatrices" begin
    trials = 100

    for m = 1:40, H = 1:4
        for n = 1:40
            A = dropzeros!(sprand(Int, m, n, 0.5))
            C = areacount(A, H = H)
            S = areasum(A, H = H)
            @testset "H = $H, m = $m" begin
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
            RC = SparseCountedRooks(N, copy(idx), H = H)
            RS = SparseSummedRooks(N, copy(idx), B.nzval, H = H)
            @testset "H = $H, N = $N" begin
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