abstract type AbstractPrimaryEdgeCutModel end

@inline (mdl::AbstractPrimaryEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl(n_vertices, n_local_pins, n_remote_pins)

struct AffinePrimaryEdgeCutModel{Tv} <: AbstractPrimaryEdgeCutModel
    α::Tv
    β_vertex::Tv
    β_local_pin::Tv
    β_remote_pin::Tv
end

@inline cost_type(::Type{AffinePrimaryEdgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffinePrimaryEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl.α + n_vertices * mdl.β_vertex + n_local_pins * mdl.β_local_pin + n_remote_pins * mdl.β_remote_pin

struct PrimaryEdgeCutOracle{Ti, Lcp, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    lcp::Lcp
    mdl::Mdl
end

oracle_model(ocl::PrimaryEdgeCutOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::PrimaryEdgeCutOracle{<:Any, <:Any, <:Any, <:AffinePrimaryEdgeCutModel})
    return bound_stripe(A, K, ocl.mdl)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffinePrimaryEdgeCutModel)
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        mdl = oracle_model(ocl)
        c_hi = mdl.α + mdl.β_vertex * n + max(mdl.β_local_pin, mdl.β_remote_pin) * N
        c_lo = mdl.α + fld(mdl.β_vertex * n + min(mdl.β_local_pin, mdl.β_remote_pin) * N, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractPrimaryEdgeCutModel, A::SparseMatrixCSC, Π; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        args, Ap = partwise(A, convert(MapPartition, Π))
        lcp = partwisecount!(hint, args..., pincount(hint, Ap, kwargs...); kwargs...)
        return PrimaryEdgeCutOracle(pos, lcp, mdl)
    end
end

@inline function (cst::PrimaryEdgeCutOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        l = cst.lcr(j, j′, k)
        return cst.mdl(j′ - j, l, w - l, k)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k) where {Ti, Mdl, Ocl <: PrimaryEdgeCutOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        w = cst.pos[j′] - cst.pos[j]
        l = Step(cst.lcp)(_j, _j′, _k)
        return cst.mdl(j′ - j, l, w - l, k)
    end
end