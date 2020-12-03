struct FunkyNetCostModel{Tv} <: AbstractNetCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

(mdl::FunkyNetCostModel)(x_width, x_work, x_net, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

@inline ChainPartitioners.cost_type(::Type{FunkyNetCostModel{Tv}}) where {Tv} = Tv

struct FunkySymCostModel{Tv} <: AbstractSymCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_net::Tv
    Δ_work::Tv
end

@inline ChainPartitioners.cost_type(::Type{FunkySymCostModel{Tv}}) where {Tv} = Tv

(mdl::FunkySymCostModel)(x_width, x_work, x_net, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

struct FunkyCommCostModel{Tv} <: AbstractCommCostModel
    α::Vector{Tv}
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

@inline ChainPartitioners.cost_type(::Type{FunkyCommCostModel{Tv}}) where {Tv} = Tv

(mdl::FunkyCommCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α[k] + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, mdl::Union{FunkyNetCostModel, FunkySymCostModel})
    ocl = oracle_stripe(mdl, A)
    m, n = size(A)
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, Π, mdl::Union{FunkyCommCostModel})
    ocl = oracle_stripe(mdl, A, Π)
    m, n = size(A)
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

LazyBisectCost = Union{AbstractNetCostModel, AbstractSymCostModel, AbstractCommCostModel}

@testset "Partitioners" begin
    for A in [
        [
            matrices["LPnetlib/lpi_itest6"],
        ];
        reshape([sprand(m, n, 0.1) for m = [1, 2, 3, 4, 8], n = [1, 2, 3, 4, 8], trial = 1:4], :);
    ]
        (m, n) = size(A)
        for K = [1, 2, 3, 4, 8]
            for f = [
                AffineWorkCostModel(0, 10, 1);
                AffineNetCostModel(0, 3, 1, 3);
                AffineCommCostModel(0, 2, 1, 3, 6);
                m == n ? AffineSymCostModel(0, 3, 1, 3, 5) : [];
                FunkyNetCostModel(rand(1:10, K), 3, 1, 3);
                FunkyCommCostModel(rand(1:10, K), 2, 1, 3, 6);
                m == n ? FunkySymCostModel(rand(1:10, K), 3, 1, 3, 5) : [];
            ]
                Π = partition_stripe(A', K, EquiSplitter())
                Φ = partition_stripe(A, K, DynamicBottleneckSplitter(f), Π)
                c = bottleneck_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (DynamicBottleneckSplitter(f), 0);
                    (BisectIndexBottleneckSplitter(f), 0);
                    (BisectCostBottleneckSplitter(f, 0.1), 0.1);
                    f isa LazyBisectCost ? (LazyBisectCostBottleneckSplitter(f, 0.1), 0.1) : [];
                    (BisectCostBottleneckSplitter(f, 0.01), 0.01);
                    f isa LazyBisectCost ? (LazyBisectCostBottleneckSplitter(f, 0.01), 0.01) : [];
                ]
                    Φ′ = partition_stripe(A, K, method, Π)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test Φ′.K == K
                    @test bottleneck_value(A, Π, Φ′, f) <= bottleneck_value(A, Π, Φ, f) * (1 + ϵ)
                end
            end

            # Creating nonnegative monotonic decreasing cost functions requires some care.
            # These complicated affine terms are designed to ensure nonnegativity.
            # Affine cost functions are commented out because their upper and lower bounds assume nonnegative coefficients.
            for f = [
                #AffineWorkCostModel(1 + nnz(A), 0, -1);
                #AffineNetCostModel(1 + nnz(A) + 3n + 3m, -3, -1, -3);
                #AffineCommCostModel(1 + nnz(A) + 3n + 6m, -2, -1, -3, -6);
                AffineLocalCostModel(0, 2, 1, 3, 6);
                FunkyNetCostModel(1 + nnz(A) + 3n + 3m .+ rand(1:10, K), -3, -1, -3);
                FunkyCommCostModel(1 + nnz(A) + 3n + 6m .+ rand(1:10, K), -2, -1, -3, -6);
                #m == n ? AffineSymCostModel(1 + nnz(A) + 18n + 3m, -3, -1, -3, 5) : [];
                m == n ? FunkySymCostModel(1 + nnz(A) + 18n + 3m .+ rand(1:10, K), -3, -1, -3, 5) : [];
            ]
                Π = partition_stripe(A', K, EquiSplitter())
                Φ = partition_stripe(A, K, DynamicBottleneckSplitter(f), Π)
                c = bottleneck_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (DynamicBottleneckSplitter(f), 0);
                    (FlipBisectIndexBottleneckSplitter(f), 0);
                    (FlipBisectCostBottleneckSplitter(f, 0.1), 0.1);
                    (FlipBisectCostBottleneckSplitter(f, 0.01), 0.01);
                    (FlipBisectCostBottleneckSplitter(f, 0.001), 0.001);
                    (FlipBisectCostBottleneckSplitter(f, 0.0001), 0.0001);
                ]
                    Φ′ = partition_stripe(A, K, method, Π)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test Φ′.K == K
                    @test bottleneck_value(A, Π, Φ′, f) <= bottleneck_value(A, Π, Φ, f) * (1 + ϵ)
                end
            end

            for f = [
                AffineNetCostModel(0, 3, 1, 3);
                m == n ? AffineSymCostModel(0, 3, 1, 3, 5) : [];
            ]
                Π = partition_stripe(A', K, EquiSplitter())
                Φ = partition_stripe(A, K, DynamicTotalSplitter(f), Π)
                c = total_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (DynamicTotalSplitter(f), 0);
                ]
                    Φ′ = partition_stripe(A, K, method, Π)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test Φ′.K == K
                    @test total_value(A, Π, Φ′, f) <= total_value(A, Π, Φ, f) * (1 + ϵ)
                end
            end

            for (f, w_max) = [
                (AffineNetCostModel(0, 3, 1, 3), 4);
            ]
                Π = partition_stripe(A', K, EquiChunker(2))
                Φ = pack_stripe(A, DynamicTotalChunker(f, w_max), Π)
                c = total_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (DynamicTotalChunker(f, w_max), 0);
                ]
                    Φ′ = pack_stripe(A, method, Π)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test total_value(A, Π, Φ′, f) <= total_value(A, Π, Φ, f) * (1 + ϵ)
                end
            end
        end
    end
end