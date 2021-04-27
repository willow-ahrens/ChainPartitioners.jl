
struct KaHyParPartitioner
    configuration::Union{Nothing, Symbol, String}
    imbalance::Union{Nothing, Float64}
    weight::Any
    verbose::Bool
end

@enum KaHyParConfiguration begin
    kahypar_configuration_communication_volume = 1
    kahypar_configuration_hyperedge_cut = 2
end

function KaHyParPartitioner(;
    configuration::Union{Nothing, KaHyParConfiguration, String}=nothing,
    imbalance::Union{Nothing, Float64} = nothing,
    weight::Any = VertexCount(),
    verbose = false
)
    if isa(configuration, KaHyParConfiguration)
        if configuration == kahypar_configuration_communication_volume
            configuration = :connectivity
        elseif configuration == kahypar_configuration_hyperedge_cut
            configuration = :edgecut
        end
    end

    return KaHyParPartitioner(
        configuration,
        imbalance,
        weight,
        verbose,
    )
end

function partition_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, K, method::KaHyParPartitioner; kwargs...) where {Tv, Ti}
    return partition_stripe(SparseMatrixCSC(A), K, method)
    #TODO gotta add the column permutation stuff
end

function partition_stripe(A::SparseMatrixCSC, K, method::KaHyParPartitioner; kwargs...)
    m, n = size(A)
    if method.imbalance === nothing
        imbalance = ()
    else
        imbalance = (:imbalance=>method.imbalance,)
    end
    if method.verbose
        asg = KaHyPar.partition(
            KaHyPar.HyperGraph(
                adjointpattern(A),
                compute_weight(A, method.weight),
                ones(Int64, m)
            ),
            K;
            configuration=method.configuration,
            imbalance...
        )
    else
        asg = @suppress KaHyPar.partition(
            KaHyPar.HyperGraph(
                adjointpattern(A),
                compute_weight(A, method.weight),
                ones(Int64, m)
            ),
            K;
            configuration=method.configuration,
            imbalance...
        )
    end
    return MapPartition(K, asg .+ true)
end
