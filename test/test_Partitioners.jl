struct FunkyNetCostModel{Tv} <: AbstractNetCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

(mdl::FunkyNetCostModel)(x_width, x_work, x_net, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

struct FunkySymCostModel{Tv} <: AbstractSymCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_net::Tv
    Δ_work::Tv
end

(mdl::FunkySymCostModel)(x_width, x_work, x_net, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

struct FunkyCommCostModel{Tv} <: AbstractCommCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

(mdl::FunkyCommCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, mdl::Union{FunkyNetCostModel, FunkySymCostModel})
    ocl = oracle_stripe(mdl, A, K)
    m, n = size(A)
    args = (maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return minmax(args...)
end

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, Π, mdl::Union{FunkyCommCostModel})
    ocl = oracle_stripe(mdl, A, K, Π)
    m, n = size(A)
    args = (maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return minmax(args...)
end

@testset "Partitioners" begin
    A = sparse([1, 1,  2, 2, 4, 8, 8,  9, 10, 10, 11, 12, 12, 12, 20, 24, 24, 24],
                [1, 12, 2, 3, 1, 3, 13, 8, 2,  3,  4,  12, 14, 16, 26, 14, 24, 15],
                [1, 1,  1, 1, 1, 1, 1,  1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ])
    f = AffineWorkCostModel(0, 10, 1)
    Φ = partition_stripe(A, 4, DynamicPartitioner(f))
    Φ′ = partition_stripe(A, 4, NicolPartitioner(f))
    @test bottleneck_stripe(A, 4, Φ, f) == bottleneck_stripe(A, 4, Φ′, f)

    for n = 1:20
        for m = 1:20
            A = sprand(m, n, 0.1)
            for k = 1:n
                Φ = partition_stripe(A, k, DynamicPartitioner(f))
                Φ′ = partition_stripe(A, k, NicolPartitioner(f))
                @test issorted(Φ.spl)
                @test Φ.spl[1] == 1
                @test Φ.spl[end] == n + 1
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test bottleneck_stripe(A, k, Φ′, f) == bottleneck_stripe(A, k, Φ, f)
                _Φ = partition_stripe(A, k, LeftistPartitioner(f))
                @test bottleneck_stripe(A, k, _Φ, f) == bottleneck_stripe(A, k, Φ, f)
                @test Φ′ == _Φ

                ϵ = 0.125
                Φ′′ = partition_stripe(A, k, BisectPartitioner(f, ϵ))
                @test issorted(Φ′′.spl)
                @test Φ′′.spl[1] == 1
                @test Φ′′.spl[end] == n + 1
                @test bottleneck_stripe(A, k, Φ′′, f) <= bottleneck_stripe(A, k, Φ, f) * (1 + ϵ)
            end
        end
    end

    f = AffineWorkCostModel(100, 0, -1)
    Φ = partition_stripe(A, 4, DynamicPartitioner(f))
    Φ′ = partition_stripe(A, 4, FlipNicolPartitioner(f))
    @test bottleneck_stripe(A, 4, Φ, f) == bottleneck_stripe(A, 4, Φ′, f)

    for n = 1:20
        for m = 1:20
            A = sprand(m, n, 0.1)
            for k = 1:n
                Φ = partition_stripe(A, k, DynamicPartitioner(f))
                Φ′ = partition_stripe(A, k, FlipNicolPartitioner(f))
                @test issorted(Φ.spl)
                @test Φ.spl[1] == 1
                @test Φ.spl[end] == n + 1
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test bottleneck_stripe(A, k, Φ′, f) == bottleneck_stripe(A, k, Φ, f)

                ϵ = 0.125
                Φ′′ = partition_stripe(A, k, FlipBisectPartitioner(f, ϵ))
                @test issorted(Φ′′.spl)
                @test Φ′′.spl[1] == 1
                @test Φ′′.spl[end] == n + 1
                @test bottleneck_stripe(A, k, Φ′′, f) <= bottleneck_stripe(A, k, Φ, f) * (1 + ϵ)
            end
        end
    end

    for n = 1:16
        for m = 1:16
            for k = 1:n
                for trial = 1:8
                    fs = Any[
                        FunkyNetCostModel(rand(2:10, k), 3, 1, 3),
                        FunkyCommCostModel(rand(2:10, k), 2, 1, 3, 6),
                    ]
                    if m == n
                        append!(fs, [
                            FunkySymCostModel(rand(2:10, k), 3, 1, 3, 5),
                        ])
                    end
                    for f = fs
                        A = sprand(m, n, 0.1)
                        Π = partition_stripe(permutedims(A), k, EquiPartitioner())
                        Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                        Φ′ = partition_stripe(A, k, NicolPartitioner(f), Π)
                        @test issorted(Φ.spl)
                        @test Φ.spl[1] == 1
                        @test Φ.spl[end] == n + 1
                        @test issorted(Φ′.spl)
                        @test Φ′.spl[1] == 1
                        @test Φ′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        _Φ = partition_stripe(A, k, LeftistPartitioner(f), Π)
                        @test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        @test Φ′ == _Φ

                        ϵ = 0.125
                        Φ′′ = partition_stripe(A, k, BisectPartitioner(f, ϵ), Π)
                        Φ′′′ = partition_stripe(A, k, LazyBisectPartitioner(f, ϵ), Π)
                        @test issorted(Φ′′.spl)
                        @test Φ′′.spl[1] == 1
                        @test Φ′′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)
                        @test Φ′′ == Φ′′′
                    end
                end
            end
        end
    end

    for n = 1:16
        for m = 1:16
            for k = 1:n
                for trial = 1:8
                    fs = Any[
                        FunkyNetCostModel(rand(-30:-22, k), 3, 1, 3),
                        FunkyCommCostModel(rand(-30:-22, k), 2, 1, 3, 6),
                    ]
                    if m == n
                        append!(fs, [
                            FunkySymCostModel(rand(-30:-22, k), 3, 1, 3, 5),
                        ])
                    end
                    for f = fs
                        A = sprand(m, n, 0.1)
                        Π = partition_stripe(permutedims(A), k, EquiPartitioner())
                        Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                        Φ′ = partition_stripe(A, k, NicolPartitioner(f), Π)
                        @test issorted(Φ.spl)
                        @test Φ.spl[1] == 1
                        @test Φ.spl[end] == n + 1
                        @test issorted(Φ′.spl)
                        @test Φ′.spl[1] == 1
                        @test Φ′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        _Φ = partition_stripe(A, k, LeftistPartitioner(f), Π)
                        @test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        @test Φ′ == _Φ
                    end
                end
            end
        end
    end

    for n = 1:16
        for m = 1:16
            for k = 1:n
                for trial = 1:8
                    fs = Any[
                        FunkyNetCostModel(rand(-10:-2, k), -3, -1, -3),
                        FunkyCommCostModel(rand(-10:-2, k), -2, -1, -3, -6),
                    ]
                    if m == n
                        append!(fs, [
                            FunkySymCostModel(rand(-10:-2, k), -3, -1, -3, 5),
                        ])
                    end
                    for f = fs
                        A = sprand(m, n, 0.1)
                        Π = partition_stripe(permutedims(A), k, EquiPartitioner())
                        Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                        Φ′ = partition_stripe(A, k, FlipNicolPartitioner(f), Π)
                        @test issorted(Φ.spl)
                        @test Φ.spl[1] == 1
                        @test Φ.spl[end] == n + 1
                        @test issorted(Φ′.spl)
                        @test Φ′.spl[1] == 1
                        @test Φ′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        _Φ = partition_stripe(A, k, FlipLeftistPartitioner(f), Π)
                        @test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        @test Φ′ == _Φ
                    end
                end
            end
        end
    end

    for n = 1:16
        for m = 1:16
            for k = 1:n
                for trial = 1:8
                    fs = Any[
                        FunkyNetCostModel((16*16 * 2) .+ rand(-10:-2, k), -3, -1, -3),
                        FunkyCommCostModel((16*16 * 2) .+ rand(-10:-2, k), -2, -1, -3, -6),
                    ]
                    if m == n
                        append!(fs, [
                            FunkySymCostModel((16*16 * 2) .+ rand(-10:-2, k), -3, -1, -3, 5),
                        ])
                    end
                    for f = fs
                        A = sprand(m, n, 0.1)
                        Π = partition_stripe(permutedims(A), k, EquiPartitioner())
                        Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                        Φ′ = partition_stripe(A, k, FlipNicolPartitioner(f), Π)
                        @test issorted(Φ.spl)
                        @test Φ.spl[1] == 1
                        @test Φ.spl[end] == n + 1
                        @test issorted(Φ′.spl)
                        @test Φ′.spl[1] == 1
                        @test Φ′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        _Φ = partition_stripe(A, k, FlipLeftistPartitioner(f), Π)
                        @test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                        @test Φ′ == _Φ

                        ϵ = 0.125
                        ϵ /= 16 * 16 * 2
                        Φ′′ = partition_stripe(A, k, FlipBisectPartitioner(f, ϵ), Π)
                        @test issorted(Φ′′.spl)
                        @test Φ′′.spl[1] == 1
                        @test Φ′′.spl[end] == n + 1
                        @test bottleneck_plaid(A, k, Π, Φ′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)
                    end
                end
            end
        end
    end
end