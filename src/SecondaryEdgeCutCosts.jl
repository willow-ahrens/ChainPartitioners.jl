abstract type AbstractSecondaryEdgeCutModel end

@inline (mdl::AbstractSecondaryEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl(n_vertices, n_local_pins, n_remote_pins)

struct AffineSecondaryEdgeCutModel{Tv} <: AbstractSecondaryEdgeCutModel
    α::Tv
    β_vertex::Tv
    β_local_pin::Tv
    β_remote_pin::Tv
end

function AffineSecondaryEdgeCutModel(; α = false, β_vertex = false, β_local_pin = false, β_remote_pin = false)
    AffineSecondaryEdgeCutModel(promote(α, β_vertex, β_local_pin, β_remote_pin)...)
end

@inline cost_type(::Type{AffineSecondaryEdgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffineSecondaryEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl.α + n_vertices * mdl.β_vertex + n_local_pins * mdl.β_local_pin + n_remote_pins * mdl.β_remote_pin

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineSecondaryEdgeCutModel)
    @assert mdl.β_vertex >= 0
    @assert mdl.β_local_pin >= 0
    @assert mdl.β_remote_pin >= 0
    adj_A = adjointpattern(A)
    c_hi = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_vertex, mdl.β_remote_pin))
    c_lo = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_vertex, mdl.β_local_pin))
    return (c_lo, c_hi)
end

struct SecondaryEdgeCutOracle{Ti, Lcv, Mdl} <: AbstractOracleCost{Mdl}
    Π::SplitPartition{Ti}
    pos::Vector{Ti}
    πos::Vector{Ti}
    lcv::Lcv
    mdl::Mdl
end

oracle_model(ocl::SecondaryEdgeCutOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, Π::SplitPartition, ocl::SecondaryEdgeCutOracle{<:Any, <:Any, <:AffineSecondaryEdgeCutModel})
    @inbounds begin
        c_lo = 0
        c_hi = 0
        (m, n) = size(A)
        mdl = oracle_model(ocl)
        @assert mdl.β_vertex >= 0
        @assert mdl.β_local_pin >= 0
        @assert mdl.β_remote_pin >= 0
        for k = 1:K
            n_vertices = Π.spl[k + 1] - Π.spl[k]
            n_pins = ocl.pos[Π.spl[k + 1]] - ocl.pos[Π.spl[k]]
            c_lo = max(c_lo, mdl.α + n_vertices * mdl.β_vertex + n_pins * min(mdl.β_local_pin, mdl.β_remote_pin))
            c_hi = max(c_hi, mdl.α + n_vertices * mdl.β_vertex + n_pins * max(mdl.β_local_pin, mdl.β_remote_pin))
        end
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractSecondaryEdgeCutModel, A::SparseMatrixCSC, Π::SplitPartition; adj_pos=nothing, adj_A=nothing, pin=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)

        if adj_A === nothing
            adj_A = adjointpattern(A)
        end

        if adj_pos === nothing
            adj_pos = adj_A.colptr
        end

        (n, K, n′, πos, prm), Ap = partwise(A, convert(MapPartition, Π))
        lcv = partwisecount!(hint, n, K, n′, πos, prm, pincount(hint, Ap); kwargs...)

        return SecondaryEdgeCutOracle(Π, adj_pos, πos, lcv, mdl)
    end
end

@inline function (cst::SecondaryEdgeCutOracle{Ti, Mdl})(i::Ti, i′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[cst.Π.spl[k + 1]] - cst.pos[cst.Π.spl[k]]
        d = cst.πos[k + 1] - cst.πos[k]
        l = cst.lcv(i, i′, k)
        return cst.mdl(cst.Π.spl[k + 1] - cst.Π.spl[k], l, w - l, k)
    end
end