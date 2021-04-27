@enum MetisScheme begin
    metis_scheme_recursive_bisection = Metis.METIS_PTYPE_RB
    metis_scheme_direct_k_way = Metis.METIS_PTYPE_KWAY
end

@enum MetisObjective begin
    metis_objective_edge_cut = Metis.METIS_OBJTYPE_CUT
    metis_objective_communication_volume = Metis.METIS_OBJTYPE_VOL
end

@enum MetisCoarsen begin
    metis_coarsen_random = Metis.METIS_CTYPE_RM
    metis_coarsen_sorted_heavy_edge = Metis.METIS_CTYPE_SHEM
end

@enum MetisInitialize begin
    metis_initialize_grow = Metis.METIS_IPTYPE_GROW
    metis_initialize_random = Metis.METIS_IPTYPE_RANDOM
    metis_initialize_edge = Metis.METIS_IPTYPE_EDGE
    metis_initialize_node = Metis.METIS_IPTYPE_NODE
end

@enum MetisRefine begin
    metis_refine_fm = Metis.METIS_RTYPE_FM
    metis_refine_greedy = Metis.METIS_RTYPE_GREEDY
    metis_refine_fm_two_sided = Metis.METIS_RTYPE_SEP2SIDED
    metis_refine_fm_one_sided = Metis.METIS_RTYPE_SEP1SIDED
end

struct MetisPartitioner
    assumesymmetry::Bool
    checksymmetry::Bool
    scheme::Union{Nothing, MetisScheme}
    objective::Union{Nothing, MetisObjective}
    coarsen::Union{Nothing, MetisCoarsen}
    initialize::Union{Nothing, MetisInitialize}
    refine::Union{Nothing, MetisRefine}
    ncuts::Union{Nothing, Int}
    niterations::Union{Nothing, Int}
    seed::Union{Nothing, Int}
    minimizeconnectivity::Union{Nothing, Bool}
    maketwohops::Union{Nothing, Bool}
    forcecontiguous::Union{Nothing, Bool}
    compressgraph::Union{Nothing, Bool}
    imbalance::Union{Nothing, Float64}
    weight::Any
    verbose_info::Bool
    verbose_time::Bool
    verbose_coarsen::Bool
    verbose_refine::Bool
    verbose_initialize::Bool
    verbose_moves::Bool
    verbose_separators::Bool
    verbose_connectivity::Bool
    verbose_contiguous::Bool
    verbose_memory::Bool
end

MetisPartitioner(;
    assumesymmetry = false,
    checksymmetry = true,
    scheme = nothing,
    objective = nothing,
    coarsen = nothing,
    initialize = nothing,
    refine = nothing,
    ncuts = nothing,
    niterations = nothing,
    seed = nothing,
    minimizeconnectivity = nothing,
    maketwohops = nothing,
    forcecontiguous = nothing,
    compressgraph = nothing,
    imbalance = nothing,
    weight = VertexCount(),
    verbose_info = false,
    verbose_time = false,
    verbose_coarsen = false,
    verbose_refine = false,
    verbose_initialize = false,
    verbose_moves = false,
    verbose_separators = false,
    verbose_connectivity = false,
    verbose_contiguous = false,
    verbose_memory = false,
) = MetisPartitioner(
    assumesymmetry,
    checksymmetry,
    scheme,
    objective,
    coarsen,
    initialize,
    refine,
    ncuts,
    niterations,
    seed,
    minimizeconnectivity,
    maketwohops,
    forcecontiguous,
    compressgraph,
    imbalance,
    weight,
    verbose_info,
    verbose_time,
    verbose_coarsen,
    verbose_refine,
    verbose_initialize,
    verbose_moves,
    verbose_separators,
    verbose_connectivity,
    verbose_contiguous,
    verbose_memory,
)

function metis_partition(A::SparseMatrixCSC, wgt, K, method::MetisPartitioner)
    old_options = copy(Metis.options)

    fill!(Metis.options, Cint(-1))
    Metis.options[Metis.METIS_OPTION_NUMBERING] = 1

    if method.scheme === nothing
        if K > 8
            alg = :KWAY
        else
            alg = :RECURSIVE
        end
    else
        if method.scheme == metis_scheme_direct_k_way
            alg = :KWAY
        elseif method.scheme == metis_scheme_recursive_bisection
            alg = :RECURSIVE
        end
    end

    if method.objective !== nothing
        Metis.options[Metis.METIS_OPTION_OBJTYPE] = Cint(method.objective)
    end
    if method.coarsen !== nothing
        Metis.options[Metis.METIS_OPTION_CTYPE] = Cint(method.coarsen)
    end
    if method.initialize !== nothing
        Metis.options[Metis.METIS_OPTION_IPTYPE] = Cint(method.initialize)
    end
    if method.refine !== nothing
        Metis.options[Metis.METIS_OPTION_RTYPE] = Cint(method.refine)
    end
    if method.ncuts !== nothing
        @assert method.ncuts >= 1
        Metis.options[Metis.METIS_OPTION_NCUTS] = Cint(method.ncuts)
    end
    if method.niterations !== nothing
        @assert method.niterations >= 0
        Metis.options[Metis.METIS_OPTION_NITER] = Cint(method.niterations)
    end
    if method.seed !== nothing
        Metis.options[Metis.METIS_OPTION_SEED] = Cint(method.seed)
    end
    if method.minimizeconnectivity !== nothing
        Metis.options[Metis.METIS_OPTION_MINCONN] = Cint(method.minimizeconnectivity)
    end
    if method.maketwohops !== nothing
        Metis.options[Metis.METIS_OPTION_NO2HOP] = Cint(!method.maketwohops)
    end
    if method.forcecontiguous !== nothing
        Metis.options[Metis.METIS_OPTION_CONTIG] = Cint(method.forcecontiguous)
    end
    if method.compressgraph !== nothing
        Metis.options[Metis.METIS_OPTION_COMPRESS] = Cint(method.compressgraph)
    end
    if method.imbalance !== nothing
        imbalance = ceil(Cint, method.imbalance * 1000)
        @assert imbalance > 0
        Metis.options[Metis.METIS_OPTION_UFACTOR] = imbalance
    end

    verbose = Cint(0)
    if method.verbose_info verbose |= Metis.METIS_DBG_INFO end
    if method.verbose_time verbose |= Metis.METIS_DBG_TIME end
    if method.verbose_coarsen verbose |= Metis.METIS_DBG_COARSEN end
    if method.verbose_refine verbose |= Metis.METIS_DBG_REFINE end
    if method.verbose_initialize verbose |= Metis.METIS_DBG_IPART end
    if method.verbose_moves verbose |= Metis.METIS_DBG_MOVEINFO end
    if method.verbose_separators verbose |= Metis.METIS_DBG_SEPINFO end
    if method.verbose_connectivity verbose |= Metis.METIS_DBG_CONNINFO end
    if method.verbose_contiguous verbose |= Metis.METIS_DBG_CONTIGINFO end
    if method.verbose_memory verbose |= Metis.METIS_DBG_MEMORY end

    g = Metis.graph(A)
    if wgt !== nothing
        g = Metis.Graph(g.nvtxs, g.xadj, g.adjncy, convert(Vector{typeof(g.nvtxs)}, wgt))
    end

    if verbose == 0
        asg = @suppress Metis.partition(A, K, alg=alg)
    else
        asg = Metis.partition(A, K, alg=alg)
    end

    Metis.options .= old_options

    return asg
end

function partition_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, K, method::MetisPartitioner; kwargs...) where {Tv, Ti}
    wgt = method.weight isa VertexCount ? nothing : compute_weight(A, method.weight)
    asg = metis_partition(pattern(SparseMatrixCSC(A)), wgt, K, method)
    return MapPartition(K, asg)
end

function partition_stripe(A::SparseMatrixCSC, K, method::MetisPartitioner; kwargs...)
    return partition_plaid(A, K, method)[end]
end

function partition_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, K, method::MetisPartitioner; kwargs...) where {Tv, Ti}
    Φ = partition_stripe(A, K, method, kwargs...)
    return (Φ, Φ)
end

function partition_plaid(A::SparseMatrixCSC, K, method::MetisPartitioner; kwargs...)
    if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
        wgt = method.weight isa VertexCount ? nothing : compute_weight(A, method.weight)
        asg = metis_partition(pattern(A), wgt, K, method)
        return (MapPartition(K, asg), MapPartition(K, asg))
    else
        m, n = size(A)
        AP = pattern(A)
        B = hvcat((2, 2), spzeros(Bool, m, m), AP, adjointpattern(AP), spzeros(Bool, n, n))
        wgt = compute_weight(A, method.weight)
        wgt = vcat(zeros(eltype(wgt), m), wgt)
        asg = metis_partition(B, wgt, K, method)
        return (MapPartition(K, asg[1:m]), MapPartition(K, asg[m + 1:end]))
    end
end