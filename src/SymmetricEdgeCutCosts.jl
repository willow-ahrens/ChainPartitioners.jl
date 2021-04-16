abstract type AbstractSymmetricEdgeCutModel end

@inline (mdl::AbstractSymmetricEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl(n_vertices, n_local_pins, n_remote_pins)

struct AffineSymmetricEdgeCutModel{Tv} <: AbstractSymmetricEdgeCutModel
    α::Tv
    β_vertex::Tv
    β_local_pin::Tv
    β_remote_pin::Tv
end

@inline cost_type(::Type{AffineSymmetricEdgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffineSymmetricEdgeCutModel)(n_vertices, n_local_pins, n_remote_pins, k) = mdl.α + n_vertices * mdl.β_vertex + n_local_pins * mdl.β_local_pin + n_remote_pins * mdl.β_remote_pin

struct SymmetricEdgeCutOracle{Ti, SelfPin, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    selfpin::SelfPin
    mdl::Mdl
end

oracle_model(ocl::SymmetricEdgeCutOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::SymmetricEdgeCutOracle{<:Any, <:Any, <:Any, <:AffineSymmetricEdgeCutModel})
    return bound_stripe(A, K, ocl.mdl)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineSymmetricEdgeCutModel)
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        mdl = oracle_model(ocl)
        c_hi = mdl.α + mdl.β_vertex * n + max(mdl.β_local_pin, mdl.β_remote_pin) * N
        c_lo = mdl.α + fld(mdl.β_vertex * n + min(mdl.β_local_pin, mdl.β_remote_pin) * N, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractSymmetricEdgeCutModel, A::SparseMatrixCSC; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        selfpin = selfpincount!(hint, A; kwargs...)
        return SymmetricEdgeCutOracle(pos, lcp, mdl)
    end
end

@inline function (cst::SymmetricEdgeCutOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        l = cst.selfpin(j, j′)
        return cst.mdl(j′ - j, l, w - l, k)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k) where {Ti, Mdl, Ocl <: SymmetricEdgeCutOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        w = cst.pos[j′] - cst.pos[j]
        l = Step(cst.selfpin)(_j, _j′)
        return cst.mdl(j′ - j, l, w - l, k)
    end
end