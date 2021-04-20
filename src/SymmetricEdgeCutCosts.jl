abstract type AbstractSymmetricEdgeCutModel end

@inline (mdl::AbstractSymmetricEdgeCutModel)(n_vertices, n_self_pins, n_cut_pins, k) = mdl(n_vertices, n_self_pins, n_cut_pins)

struct AffineSymmetricEdgeCutModel{Tv} <: AbstractSymmetricEdgeCutModel
    α::Tv
    β_vertex::Tv
    β_self_pin::Tv
    β_cut_pin::Tv
end

function AffineSymmetricEdgeCutModel(; α = false, β_vertex = false, β_self_pin = false, β_cut_pin = false)
    AffineSymmetricEdgeCutModel(promote(α, β_vertex, β_self_pin, β_cut_pin)...)
end

@inline cost_type(::Type{AffineSymmetricEdgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffineSymmetricEdgeCutModel)(n_vertices, n_self_pins, n_cut_pins, k) = mdl.α + n_vertices * mdl.β_vertex + n_self_pins * mdl.β_self_pin + n_cut_pins * mdl.β_cut_pin

struct SymmetricEdgeCutOracle{Ti, SelfPin, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    selfpin::SelfPin
    mdl::Mdl
end

oracle_model(ocl::SymmetricEdgeCutOracle) = ocl.mdl

function oracle_stripe(hint::AbstractHint, mdl::AbstractSymmetricEdgeCutModel, A::SparseMatrixCSC; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        selfpin = selfpincount(hint, A; kwargs...)
        return SymmetricEdgeCutOracle(pos, selfpin, mdl)
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