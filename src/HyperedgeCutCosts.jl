abstract type AbstractHyperedgeCutModel end

@inline (mdl::AbstractHyperedgeCutModel)(n_vertices, n_pins, n_self_nets, n_cut_nets, k) = mdl(n_vertices, n_pins, n_self_nets, n_cut_nets)



struct AffineHyperedgeCutModel{Tv} <: AbstractHyperedgeCutModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_self_net::Tv
    β_cut_net::Tv
end

function AffineHyperedgeCutModel(; α = false, β_vertex = false, β_pin = false, β_self_net = false, β_cut_net = false)
    AffineHyperedgeCutModel(promote(α, β_vertex, β_pin, β_self_net, β_cut_net)...)
end

@inline cost_type(::Type{AffineHyperedgeCutModel{Tv}}) where {Tv} = Tv

(mdl::AffineHyperedgeCutModel)(n_vertices, n_pins, n_self_nets, n_cut_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_self_nets * mdl.β_self_net + n_cut_nets * mdl.β_cut_net

struct HyperedgeCutOracle{Ti, Net, SelfNet, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    selfnet::SelfNet
    mdl::Mdl
end

oracle_model(ocl::HyperedgeCutOracle) = ocl.mdl

function oracle_stripe(hint::AbstractHint, mdl::AbstractHyperedgeCutModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = netcount(hint, A; kwargs...)
        end
        selfnet = selfnetcount(hint, A; kwargs...)
        return HyperedgeCutOracle(pos, net, selfnet, mdl)
    end
end

@inline function (cst::HyperedgeCutOracle{Ti, Mdl})(j, j′, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net[j, j′]
        l = cst.selfnet[j, j′]
        return cst.mdl(j′ - j, w, l, d - l, k...)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k...) where {Ti, Mdl, Ocl <: HyperedgeCutOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = maptuple(destep, _k...)
        w = cst.pos[j′] - cst.pos[j]
        d = Step(cst.net)(_j, _j′)
        l = Step(cst.selfnet)(_j, _j′)
        return cst.mdl(j′ - j, w, l, d - l, k...)
    end
end