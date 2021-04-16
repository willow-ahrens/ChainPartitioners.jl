abstract type AbstractSecondaryConnectivityModel end

@inline (mdl::AbstractSecondaryConnectivityModel)(x_width, x_work, x_net, x_local, k) = mdl(x_width, x_work, x_net, x_local)

struct AffineSecondaryConnectivityModel{Tv} <: AbstractSecondaryConnectivityModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

@inline cost_type(::Type{AffineSecondaryConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffineSecondaryConnectivityModel)(x_width, x_work, x_local, x_comm, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineSecondaryConnectivityModel)
    adj_A = adjointpattern(A)
    c_hi = bottleneck_value(adj_A, Π, AffineConnectivityModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_comm))
    c_lo = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_width, mdl.β_work))
    return (c_lo, c_hi)
end

struct SecondaryConnectivityOracle{Ti, Lcc, Mdl} <: AbstractOracleCost{Mdl}
    Π::SplitPartition{Ti}
    pos::Vector{Ti}
    πos::Vector{Ti}
    lcc::Lcc
    mdl::Mdl
end

oracle_model(ocl::SecondaryConnectivityOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, Π::SplitPartition, ocl::SecondaryConnectivityOracle{<:Any, <:Any, <:AffineSecondaryConnectivityModel})
    @inbounds begin
        c_lo = 0
        c_hi = 0
        (m, n) = size(A)
        mdl = oracle_model(ocl)
        for k = 1:K
            x_width = Π.spl[k + 1] - Π.spl[k]
            x_work = ocl.pos[Π.spl[k + 1]] - ocl.pos[Π.spl[k]]
            x_comm = ocl.πos[k + 1] - ocl.πos[k]
            c_lo = max(c_lo, mdl.α + x_width * mdl.β_width + x_work * mdl.β_work)
            c_hi = max(c_hi, mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_comm * mdl.β_comm)
        end
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractSecondaryConnectivityModel, A::SparseMatrixCSC, Π::SplitPartition; adj_pos=nothing, adj_A=nothing, net=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)

        if adj_A === nothing
            adj_A = adjointpattern(A)
        end

        if adj_pos === nothing
            adj_pos = adj_A.colptr
        end

        (n, K, n′, πos, prm), _ = partwise(A, convert(MapPartition, Π))
        lcc = partwisecost!(hint, n, K, n′, πos, prm, VertexCount(); kwargs...)

        return SecondaryConnectivityOracle(Π, adj_pos, πos, lcc, mdl)
    end
end

@inline function (cst::SecondaryConnectivityOracle{Ti, Mdl})(i::Ti, i′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[cst.Π.spl[k + 1]] - cst.pos[cst.Π.spl[k]]
        d = cst.πos[k + 1] - cst.πos[k]
        l = cst.lcc(i, i′, k)
        return cst.mdl(cst.Π.spl[k + 1] - cst.Π.spl[k], w, l, d - l, k)
    end
end

function compute_objective(g::G, A, Π, Φ::SplitPartition, mdl::AffineSecondaryConnectivityModel) where {G}
    return compute_objective(g, adjointpattern(A), Φ, Π, AffinePrimaryConnectivityModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end
function compute_objective(g::G, A, Π, Φ, mdl::AffineSecondaryConnectivityModel) where {G}
    return compute_objective(g, adjointpattern(A), Φ, Π, AffinePrimaryConnectivityModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end