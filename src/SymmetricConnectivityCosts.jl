abstract type AbstractSymmetricConnectivityModel end

@inline (mdl::AbstractSymmetricConnectivityModel)(n_vertices, n_pins, n_local_nets, n_remote_nets, k) = mdl(n_vertices, n_pins, n_local_nets, n_remote_nets)

struct AffineSymmetricConnectivityModel{Tv} <: AbstractSymmetricConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_local_net::Tv
    β_remote_net::Tv
end

function AffineSymmetricConnectivityModel(; α = false, β_vertex = false, β_pin = false, β_local_net = false, β_remote_net = false)
    AffineSymmetricConnectivityModel(promote(α, β_vertex, β_pin, β_local_net, β_remote_net)...)
end

@inline cost_type(::Type{AffineSymmetricConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffineSymmetricConnectivityModel)(n_vertices, n_pins, n_local_nets, n_remote_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_local_nets * mdl.β_local_net + n_remote_nets * mdl.β_remote_net

struct SymmetricConnectivityOracle{Ti, Net, DiaNet, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    dianet::DiaNet
    mdl::Mdl
end

oracle_model(ocl::SymmetricConnectivityOracle) = ocl.mdl

function oracle_stripe(hint::AbstractHint, mdl::AbstractSymmetricConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        @assert m == n
        N = nnz(A)
        pos = A.colptr

        if net === nothing
            net = netcount(hint, A)
        end

        dianet = dianetcount(hint, A)

        return SymmetricConnectivityOracle(pos, net, dianet, mdl)
    end
end

@inline function (cst::SymmetricConnectivityOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net(j, j′)
        r = cst.dianet(j, j′) - (j′ - j)
        l = d - r
        return cst.mdl(j′ - j, w, l, r, k...)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k...) where {Ti, Mdl, Ocl <: SymmetricConnectivityOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = maptuple(destep, _k...)
        w = cst.pos[j′] - cst.pos[j]
        d = Step(cst.net)(_j, _j′)
        r = Step(cst.dianet)(_j, _j′) - (j′ - j)
        l = d - r
        return cst.mdl(j′ - j, w, l, r, k...)
    end
end