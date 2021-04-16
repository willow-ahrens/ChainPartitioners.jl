abstract type AbstractSecondaryConnectivityModel end

@inline (mdl::AbstractSecondaryConnectivityModel)(n_vertices, n_pins, n_nets, n_local_nets, k) = mdl(n_vertices, n_pins, n_nets, n_local_nets)

struct AffineSecondaryConnectivityModel{Tv} <: AbstractSecondaryConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_local_net::Tv
    β_remote_net::Tv
end

@inline cost_type(::Type{AffineSecondaryConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffineSecondaryConnectivityModel)(n_vertices, n_pins, n_local_nets, n_remote_nets, k) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_local_nets * mdl.β_local_net + n_remote_nets * mdl.β_remote_net

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineSecondaryConnectivityModel)
    adj_A = adjointpattern(A)
    c_hi = bottleneck_value(adj_A, Π, AffineConnectivityModel(mdl.α, mdl.β_vertex, mdl.β_pin, mdl.β_remote_net))
    c_lo = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_vertex, mdl.β_pin))
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
            n_vertices = Π.spl[k + 1] - Π.spl[k]
            n_pins = ocl.pos[Π.spl[k + 1]] - ocl.pos[Π.spl[k]]
            n_remote_nets = ocl.πos[k + 1] - ocl.πos[k]
            c_lo = max(c_lo, mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin)
            c_hi = max(c_hi, mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_remote_nets * mdl.β_remote_net)
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
        lcc = partwisecount!(hint, n, K, n′, πos, prm, VertexCount(); kwargs...)

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
    return compute_objective(g, adjointpattern(A), Φ, Π, AffinePrimaryConnectivityModel(mdl.α, mdl.β_vertex, mdl.β_pin, mdl.β_local_net, mdl.β_remote_net))
end
function compute_objective(g::G, A, Π, Φ, mdl::AffineSecondaryConnectivityModel) where {G}
    return compute_objective(g, adjointpattern(A), Φ, Π, AffinePrimaryConnectivityModel(mdl.α, mdl.β_vertex, mdl.β_pin, mdl.β_local_net, mdl.β_remote_net))
end