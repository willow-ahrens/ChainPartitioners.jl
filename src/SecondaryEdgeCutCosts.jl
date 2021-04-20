abstract type AbstractSecondaryEdgeCutModel end

@inline (mdl::AbstractSecondaryEdgeCutModel)(n_vertices, n_self_pins, n_cut_pins, k) = mdl(n_vertices, n_self_pins, n_cut_pins)

struct AffineSecondaryEdgeCutModel{Tv} <: AbstractSecondaryEdgeCutModel
    α::Tv
    β_vertex::Tv
    β_self_pin::Tv
    β_cut_pin::Tv
end

function AffineSecondaryEdgeCutModel(; α = false, β_vertex = false, β_self_pin = false, β_cut_pin = false)
    AffineSecondaryEdgeCutModel(promote(α, β_vertex, β_self_pin, β_cut_pin)...)
end

@inline cost_type(::Type{AffineSecondaryEdgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffineSecondaryEdgeCutModel)(n_vertices, n_self_pins, n_cut_pins, k) = mdl.α + n_vertices * mdl.β_vertex + n_self_pins * mdl.β_self_pin + n_cut_pins * mdl.β_cut_pin

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineSecondaryEdgeCutModel)
    @assert mdl.β_vertex >= 0
    @assert mdl.β_self_pin >= 0
    @assert mdl.β_cut_pin >= 0
    adj_A = adjointpattern(A)
    c_hi = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_vertex, mdl.β_cut_pin))
    c_lo = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_vertex, mdl.β_self_pin))
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
        @assert mdl.β_self_pin >= 0
        @assert mdl.β_cut_pin >= 0
        for k = 1:K
            n_vertices = Π.spl[k + 1] - Π.spl[k]
            n_pins = ocl.pos[Π.spl[k + 1]] - ocl.pos[Π.spl[k]]
            c_lo = max(c_lo, mdl.α + n_vertices * mdl.β_vertex + n_pins * min(mdl.β_self_pin, mdl.β_cut_pin))
            c_hi = max(c_hi, mdl.α + n_vertices * mdl.β_vertex + n_pins * max(mdl.β_self_pin, mdl.β_cut_pin))
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