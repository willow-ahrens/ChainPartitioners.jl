using ChainPartitioners: destep

function Δ(i, is) 
    moves = [ChainPartitioners.Same(i), ChainPartitioners.Same(i), ChainPartitioners.Jump(rand(is))]
    if i - 1 in is
        push!(moves, ChainPartitioners.Prev(i - 1))
    end
    if i + 1 in is
        push!(moves, ChainPartitioners.Next(i + 1))
    end
    return moves
end

ref_dominancecount(A, i, j) = sum(A[1:i - 1, 1:j - 1] .!= 0)
ref_dominancesum(A, i, j) = sum(A[1:i - 1, 1:j - 1])

test_points(trials, Is...) = collect(Iterators.flatten((
    Iterators.product(map(I -> (first(I), last(I)), Is)...),
    zip(map(I -> rand(I, trials), Is)...)
)))[randperm(end)]

test_dims = [1:16..., 31, 32, 33, 63, 64, 65]

test_dims = [1:3..., 7, 8, 9]

@testset "SparsePrefixMatrices" begin
    @testset "jump" begin
        for (hint, kwargs) = (
            ((), ()),
            ((), (H = 1,)),
            ((ChainPartitioners.SparseHint(),), (H = 1,)),
            ((ChainPartitioners.SparseHint(),), (H = 2,)),
            ((ChainPartitioners.SparseHint(),), (H = 3,)),
            ((ChainPartitioners.SparseHint(),), (H = 4,)),
            ((ChainPartitioners.SparseHint(),), (b = 1,)),
            ((ChainPartitioners.SparseHint(),), (b = 2,)),
            ((ChainPartitioners.SparseHint(),), (b = 3,)),
            ((ChainPartitioners.SparseHint(),), (b = 4,)),
            ((ChainPartitioners.RandomHint(),), ()),
            ((), (b = 1,)),
        )
            for m = test_dims
                for n = test_dims
                    A = dropzeros!(sprand(UInt, m, n, 0.5))

                    C = dominancecount(hint..., A; kwargs...)
                    S = dominancesum(hint..., A; kwargs...)
                    @testset "kwargs = $kwargs, m = $m" begin
                        for (i, j) in test_points(10, 1:m + 1, 1:n + 1)
                            @test C[i, j] == ref_dominancecount(A, i, j)
                            @test S[i, j] == ref_dominancesum(A, i, j)
                        end
                    end
                end

                N = m
                idx = randperm(N)
                B = spzeros(UInt, N, N)
                for j = 1:N
                    B[idx[j], j] = (rand(UInt) << 1) + 1
                end
                RC = rookcount!(hint..., N, copy(idx); kwargs...)
                RS = rooksum!(hint..., N, copy(idx), B.nzval; kwargs...)
                @testset "kwargs = $kwargs, N = $N" begin
                    for (i, j) in test_points(100, 1:N + 1, 1:N + 1)
                        @test RC[i, j] == ref_dominancecount(B, i, j)
                        @test RS[i, j] == ref_dominancesum(B, i, j)
                    end
                end
            end
        end
    end

    @testset "step" begin
        for m = 1:40
            for n = 1:40
                A = dropzeros!(sprand(Int, m, n, 0.5))
                C = dominancecount(ChainPartitioners.StepHint(), A)
                @testset "m = $m" begin
                    for (i, j) = test_points(10, 1:m + 1, 1:n + 1)
                        @test C[i, j] == sum(A[1:i - 1, 1:j - 1] .!= 0)

                        for _ = 1:8
                            (_i, _j) = (rand(Δ(i, 1:m + 1)), rand(Δ(j, 1:n + 1)))
                            (i, j) = map(destep, (_i, _j))
                            @test ChainPartitioners.Step(C)(_i, _j) == ref_dominancecount(A, i, j)
                        end
                    end
                end
            end
        end
    end
end