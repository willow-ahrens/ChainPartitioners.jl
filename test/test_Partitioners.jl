struct FunkyConnectivityModel{Tv} <: AbstractConnectivityModel
    α::Vector{Tv}
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
end

(mdl::FunkyConnectivityModel)(n_vertices, n_pins, n_nets, k) = mdl.α[k] + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net 

@inline ChainPartitioners.cost_type(::Type{FunkyConnectivityModel{Tv}}) where {Tv} = Tv

struct FunkyMonotonizedSymmetricConnectivityModel{Tv} <: AbstractMonotonizedSymmetricConnectivityModel
    α::Vector{Tv}
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
    Δ_pins::Tv
end

@inline ChainPartitioners.cost_type(::Type{FunkyMonotonizedSymmetricConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::FunkyMonotonizedSymmetricConnectivityModel)(n_vertices, n_pins, n_nets, k) = mdl.α[k] + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net 

struct FunkyPrimaryConnectivityModel{Tv} <: AbstractPrimaryConnectivityModel
    α::Vector{Tv}
    β_vertex::Tv
    β_pin::Tv
    β_local_net::Tv
    β_remote_net::Tv
end

@inline ChainPartitioners.cost_type(::Type{FunkyPrimaryConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::FunkyPrimaryConnectivityModel)(n_vertices, n_pins, n_local_nets, n_remote_nets, k) = mdl.α[k] + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_local_nets * mdl.β_local_net + n_remote_nets * mdl.β_remote_net

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, mdl::Union{FunkyConnectivityModel, FunkyMonotonizedSymmetricConnectivityModel})
    ocl = oracle_stripe(mdl, A)
    m, n = size(A)
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

function ChainPartitioners.bound_stripe(A::SparseMatrixCSC, K, Π, mdl::Union{FunkyPrimaryConnectivityModel})
    ocl = oracle_stripe(mdl, A, Π)
    m, n = size(A)
    args = (minimum(mdl.α), maximum(mdl.α), maximum(ocl(1, n + 1, k) for k = 1:K))
    return (minimum(args), maximum(args))
end

LazyBisectCost = Union{AbstractConnectivityModel, AbstractMonotonizedSymmetricConnectivityModel, AbstractPrimaryConnectivityModel}



struct ConvexWorkCostModel <: AbstractWorkCostModel
    α::Float64
    β_vertex::Float64
    β_pin::Float64
end

(mdl::ConvexWorkCostModel)(n_vertices, n_pins) = mdl.α + (n_vertices * mdl.β_vertex + n_pins * mdl.β_pin)^0.8

@inline ChainPartitioners.cost_type(::Type{ConvexWorkCostModel}) = Float64

struct ConcaveWorkCostModel <: AbstractWorkCostModel
    α::Float64
    β_vertex::Float64
    β_pin::Float64
end

(mdl::ConcaveWorkCostModel)(n_vertices, n_pins) = mdl.α + (n_vertices * mdl.β_vertex + n_pins * mdl.β_pin)^2

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
                AffineConnectivityModel(0, 3, 1, 3);
                AffinePrimaryConnectivityModel(0, 2, 1, 3, 6);
                m == n ? AffineMonotonizedSymmetricConnectivityModel(0, 3, 1, 3, 5) : [];
                FunkyConnectivityModel(rand(1:10, K), 3, 1, 3);
                FunkyPrimaryConnectivityModel(rand(1:10, K), 2, 1, 3, 6);
                m == n ? FunkyMonotonizedSymmetricConnectivityModel(rand(1:10, K), 3, 1, 3, 5) : [];
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
                #AffineConnectivityModel(1 + nnz(A) + 3n + 3m, -3, -1, -3);
                #AffinePrimaryConnectivityModel(1 + nnz(A) + 3n + 6m, -2, -1, -3, -6);
                AffineSecondaryConnectivityModel(0, 2, 1, 3, 6);
                FunkyConnectivityModel(1 + nnz(A) + 3n + 3m .+ rand(1:10, K), -3, -1, -3);
                FunkyPrimaryConnectivityModel(1 + nnz(A) + 3n + 6m .+ rand(1:10, K), -2, -1, -3, -6);
                #m == n ? AffineMonotonizedSymmetricConnectivityModel(1 + nnz(A) + 18n + 3m, -3, -1, -3, 5) : [];
                m == n ? FunkyMonotonizedSymmetricConnectivityModel(1 + nnz(A) + 18n + 3m .+ rand(1:10, K), -3, -1, -3, 5) : [];
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
                AffineConnectivityModel(0, 3, 1, 3);
                m == n ? AffineMonotonizedSymmetricConnectivityModel(0, 3, 1, 3, 5) : [];
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
                (AffineConnectivityModel(0.0, 0.0, 0.0, 1.0),);
                (AffineWorkCostModel(0, 0, 0),);
                (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 2),);
                (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 4),);
                (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
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
            (ConstrainedCost(AffineConnectivityModel(0, 3, 1, 3), VertexCount(), 4), 4);
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
            (AffineConnectivityModel(-0.5, 0.0, 0.0, 1.0),);
            (AffineConnectivityModel(0, 0, 0, 1),);
            (AffineWorkCostModel(0, 0, 0),);
            (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 2),);
            (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 4),);
            (ConstrainedCost(AffineConnectivityModel(0, 0, 0, 1), AffineWorkCostModel(0, 1, 0), 8),);
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