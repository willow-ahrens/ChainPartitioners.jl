@testset "EnvelopeMatrices" begin
    trials = 100

    for m = 1:100, h = 1:4
        n = m
        A = dropzeros!(sprand(UInt, m, n, 0.5))
        env = rowenvelope(A)
        @testset "h = $h, m = $m" begin
            for _ = 1:trials
                j = rand(1:n + 1)
                j′ = rand(j:n + 1)
                if length(A.rowval[A.colptr[j]:A.colptr[j′] - 1]) > 0
                    @test env[j, j′] == extrema(A.rowval[A.colptr[j]:A.colptr[j′] - 1])
                else
                    @test env[j, j′] == (m + 1, 0)
                end
            end
        end
    end
end