@testset "util" begin
    trials = 100

    for m = 1:100, n = 1:100
        A = sprand(Bool, m, n, 0.5)
        @test ChainPartitioners.adjointpattern(A) == permutedims(A)
        A = sprand(UInt, m, n, 0.5)
        A.nzval .= true
        @test ChainPartitioners.adjointpattern(A) == permutedims(A)
    end
end