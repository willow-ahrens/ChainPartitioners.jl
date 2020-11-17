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
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, Π, mdl::Union{FunkyCommCostModel})
    ocl = oracle_stripe(mdl, A, K, Π)
    m, n = size(A)
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

@testset "Partitioners" begin
    As = vcat(
        [
            matrices["LPnetlib/lpi_itest6"],
        ],
        reshape([sprand(m, n, 0.1) for m = [1, 2, 3, 4, 8, 16], n = [1, 2, 3, 4, 8, 16], trial = 1:8], :),
    )

    for A in As
        (m, n) = size(A)

        for k = [1, 2, 3, 4, 8, 16]
            fs = Any[
                AffineWorkCostModel(0, 10, 1),
                AffineNetCostModel(0, 3, 1, 3),
                AffineCommCostModel(0, 2, 1, 3, 6),
            ]
            if m == n
                append!(fs, [
                    AffineSymCostModel(0, 3, 1, 3, 5),
                ])
            end
            funky_fs = Any[
                FunkyNetCostModel(rand(1:10, k), 3, 1, 3),
                FunkyCommCostModel(rand(1:10, k), 2, 1, 3, 6),
            ]
            if m == n
                append!(funky_fs, [
                    FunkySymCostModel(rand(1:10, k), 3, 1, 3, 5),
                ])
            end
            funky_fs = Any[
                FunkyNetCostModel(rand(1:10, k), 3, 1, 3),
                FunkyCommCostModel(rand(1:10, k), 2, 1, 3, 6),
            ]
            if m == n
                append!(funky_fs, [
                    FunkySymCostModel(rand(1:10, k), 3, 1, 3, 5),
                ])
            end

            # Creating nonnegative monotonic decreasing cost functions requires some care.
            # These complicated affine terms are designed to ensure nonnegativity.
            # Affine cost functions are commented out because their upper and lower bounds assume nonnegative coefficients.
            flip_fs = Any[
                #AffineWorkCostModel(1 + nnz(A), 0, -1),
                #AffineNetCostModel(1 + nnz(A) + 3n + 3m, -3, -1, -3),
                #AffineCommCostModel(1 + nnz(A) + 3n + 6m, -2, -1, -3, -6),
                AffineLocalCostModel(0, 2, 1, 3, 6),
                FunkyNetCostModel(1 + nnz(A) + 3n + 3m .+ rand(1:10, k), -3, -1, -3),
                FunkyCommCostModel(1 + nnz(A) + 3n + 6m .+ rand(1:10, k), -2, -1, -3, -6),
            ]
            if m == n
                append!(flip_fs, [
                    #AffineSymCostModel(1 + nnz(A) + 18n + 3m, -3, -1, -3, 5),
                    FunkySymCostModel(1 + nnz(A) + 18n + 3m .+ rand(1:10, k), -3, -1, -3, 5),
                ])
            end

            Π = partition_stripe(permutedims(A), k, EquiPartitioner())

            for f = vcat(fs, funky_fs)
                Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                Φ′ = partition_stripe(A, k, NicolPartitioner(f), Π)
                @test issorted(Φ.spl)
                @test Φ.spl[1] == 1
                @test Φ.spl[end] == n + 1
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                #_Φ = partition_stripe(A, k, LeftistPartitioner(f), Π)
                #@test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                #@test Φ′ == _Φ
            end

            for f = fs[2:end]
                Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                ϵ = 0.125
                Φ′′ = partition_stripe(A, k, BisectPartitioner(f, ϵ), Π)
                Φ′′′ = partition_stripe(A, k, LazyBisectPartitioner(f, ϵ), Π)
                @test issorted(Φ′′.spl)
                @test Φ′′.spl[1] == 1
                @test Φ′′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)
                @test Φ′′ == Φ′′′
            end

            for f = funky_fs
                Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                ϵ = 0.125
                Φ′′ = partition_stripe(A, k, BisectPartitioner(f, ϵ), Π)
                @test issorted(Φ′′.spl)
                @test Φ′′.spl[1] == 1
                @test Φ′′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)

                Φ′′′ = partition_stripe(A, k, LazyBisectPartitioner(f, ϵ), Π)
                @test issorted(Φ′′′.spl)
                @test Φ′′′.spl[1] == 1
                @test Φ′′′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)
            end

            for f in flip_fs
                Φ = partition_stripe(A, k, DynamicPartitioner(f), Π)
                Φ′ = partition_stripe(A, k, FlipNicolPartitioner(f), Π)
                @test issorted(Φ.spl)
                @test Φ.spl[1] == 1
                @test Φ.spl[end] == n + 1
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′, f) == bottleneck_plaid(A, k, Π, Φ, f)
                #_Φ = partition_stripe(A, k, FlipLeftistPartitioner(f), Π)
                #@test bottleneck_plaid(A, k, Π, _Φ, f) == bottleneck_plaid(A, k, Π, Φ, f)
                #@test Φ′ == _Φ

                ϵ = 0.0001
                Φ′′ = partition_stripe(A, k, FlipBisectPartitioner(f, ϵ), Π)
                @test issorted(Φ′′.spl)
                @test Φ′′.spl[1] == 1
                @test Φ′′.spl[end] == n + 1
                @test bottleneck_plaid(A, k, Π, Φ′′, f) <= bottleneck_plaid(A, k, Π, Φ, f) * (1 + ϵ)
            end
        end
    end
end