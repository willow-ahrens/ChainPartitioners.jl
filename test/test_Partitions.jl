@testset "Partitions" begin
    trials = 100

    for m = 1:100, K = 1:4, _ = 1:trials
        Φ1 = MapPartition(K, rand(1:K, m))
        Φ2 = convert(DomainPartition, Φ1)
        @test convert(MapPartition, Φ2) == Φ1

        Φ1 = DomainPartition(K, randperm(m), [1, sort(rand(1:(m + 1), K - 1))..., m + 1])
        for k = 1:K
            sort!(@view Φ1.prm[Φ1.spl[k] : Φ1.spl[k + 1] - 1])
        end
        Φ2 = convert(MapPartition, Φ1)
        @test convert(DomainPartition, Φ2) == Φ1
    end
end