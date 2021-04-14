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



struct ConvexWorkCostModel <: AbstractWorkCostModel
    α::Float64
    β_width::Float64
    β_work::Float64
end

(mdl::ConvexWorkCostModel)(x_width, x_work) = mdl.α + (x_width * mdl.β_width + x_work * mdl.β_work)^0.8

@inline ChainPartitioners.cost_type(::Type{ConvexWorkCostModel}) = Float64

struct ConcaveWorkCostModel <: AbstractWorkCostModel
    α::Float64
    β_width::Float64
    β_work::Float64
end

(mdl::ConcaveWorkCostModel)(x_width, x_work) = mdl.α + (x_width * mdl.β_width + x_work * mdl.β_work)^2

@inline ChainPartitioners.cost_type(::Type{ConcaveWorkCostModel}) = Float64



@testset "Partitioners" begin
    for A in [
        [
            matrices["LPnetlib/lpi_itest6"],
            SparseMatrixCSC(matrices["HB/can_292"]),
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
                Φ = partition_stripe(A, K, ReferenceBottleneckSplitter(f), Π)
                c = bottleneck_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (ReferenceBottleneckSplitter(f), 0);
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
                Φ = partition_stripe(A, K, ReferenceBottleneckSplitter(f), Π)
                c = bottleneck_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (ReferenceBottleneckSplitter(f), 0);
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
                Φ = partition_stripe(A, K, ReferenceTotalSplitter(f), Π)
                c = total_value(A, Π, Φ, f)
                for (method, ϵ) = [
                    (ReferenceTotalSplitter(f), 0);
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

            for (f,) = [
                (ConvexWorkCostModel(0, 0, 1),);
                (AffineNetCostModel(0.0, 0.0, 0.0, 1.0),);
                (AffineWorkCostModel(0, 0, 0),);
                (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 2),);
                (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 4),);
                (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
                (ConstrainedCost(ConvexWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 2),);
                (ConstrainedCost(ConvexWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 4),);
                (ConstrainedCost(ConvexWorkCostModel(0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
            ]
                Φ = partition_stripe(A, K, ReferenceTotalSplitter(f))
                c = total_value(A, Φ, f)
                for method = [
                    ReferenceTotalSplitter(f);
                    DynamicTotalSplitter(f);
                    DynamicTotalChunker(f);
                    ConvexTotalSplitter(f);
                ]
                    Φ′ = partition_stripe(A, K, method)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test total_value(A, Φ′, f) ≈ c
                end
            end

            for (f,) = [
                (ConcaveWorkCostModel(0, 0, 1),);
                (AffineWorkCostModel(0, 0, 0),);
                (ConstrainedCost(ConcaveWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 2),);
                (ConstrainedCost(ConcaveWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 4),);
                (ConstrainedCost(ConcaveWorkCostModel(0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
            ]
                Φ = partition_stripe(A, K, ReferenceTotalSplitter(f))
                c = total_value(A, Φ, f)
                for method = [
                    ReferenceTotalSplitter(f);
                    DynamicTotalSplitter(f);
                    DynamicTotalChunker(f);
                    ConcaveTotalSplitter(f);
                ]
                    Φ′ = partition_stripe(A, K, method)
                    @test issorted(Φ′.spl)
                    @test Φ′.spl[1] == 1
                    @test Φ′.spl[end] == n + 1
                    @test total_value(A, Φ′, f) ≈ c
                end
            end
        end

        for (f, w_max) = [
            (ConstrainedCost(AffineNetCostModel(0, 3, 1, 3), VertexCount(), 4), 4);
            (ConstrainedCost(BlockComponentCostModel{Int64}(0, 0, (10, identity), (2, x->2x)), VertexCount(), 4), 4);
            (ConstrainedCost(BlockComponentCostModel{Int64}(identity, x->3x, (10, identity), (2, x->2x)), VertexCount(), 4), 4);
        ]
            Π = pack_stripe(A', EquiChunker(2))
            Φ = pack_stripe(A, DynamicTotalChunker(f), Π) #The dynamic total chunker should be at least as good as optimal
            c = total_value(A, Π, Φ, f)
            for method = [
                ReferenceTotalChunker(f);
                DynamicTotalChunker(f)
                StrictChunker(w_max);
                OverlapChunker(0.9, w_max);
                OverlapChunker(0.8, w_max);
                OverlapChunker(0.7, w_max);
            ]
                Φ′ = pack_stripe(A, method, Π)
                @test issorted(Φ′.spl)
                @test all(Φ′.spl[2:end] .- Φ′.spl[1:end - 1] .<= w_max)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test total_value(A, Π, Φ′, f) >= total_value(A, Π, Φ, f)
            end
        end

        for (f,) = [
            (ConvexWorkCostModel(0.0, 0, 1),);
            (ConvexWorkCostModel(-0.7, 0, 1),);
            (AffineNetCostModel(-0.5, 0.0, 0.0, 1.0),);
            (AffineNetCostModel(0, 0, 0, 1),);
            (AffineWorkCostModel(0, 0, 0),);
            (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 2),);
            (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 4),);
            (ConstrainedCost(AffineNetCostModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
            (ConstrainedCost(ConvexWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 2),);
            (ConstrainedCost(ConvexWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 4),);
            (ConstrainedCost(ConvexWorkCostModel(0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
        ]
            Φ = pack_stripe(A, ReferenceTotalChunker(f))
            c = total_value(A, Φ, f)
            for method = [
                ReferenceTotalChunker(f);
                DynamicTotalChunker(f);
                ConvexTotalChunker(f);
            ]
                Φ′ = pack_stripe(A, method)
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test total_value(A, Φ′, f) ≈ c
            end
        end

        for (f,) = [
            (ConcaveWorkCostModel(0.0, 0, 1),);
            (ConcaveWorkCostModel(-0.7, 0, 1),);
            (AffineWorkCostModel(0, 0, 0),);
            (ConstrainedCost(ConcaveWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 2),);
            (ConstrainedCost(ConcaveWorkCostModel(0, 1, 0), AffineWorkCostModel(0, 1, 0), 4),);
            (ConstrainedCost(ConcaveWorkCostModel(0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
        ]
            Φ = pack_stripe(A, ReferenceTotalChunker(f))
            c = total_value(A, Φ, f)
            for method = [
                ReferenceTotalChunker(f);
                DynamicTotalChunker(f);
                ConcaveTotalChunker(f);
            ]
                Φ′ = pack_stripe(A, method)
                @test issorted(Φ′.spl)
                @test Φ′.spl[1] == 1
                @test Φ′.spl[end] == n + 1
                @test total_value(A, Φ′, f) ≈ c
            end
        end

    end
end