abstract type AbstractNetCostModel end

@inline (mdl::AbstractNetCostModel)(x_width, x_work, x_net, k) = mdl(x_width, x_work, x_net)



struct AffineNetCostModel{Tv} <: AbstractNetCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

@inline cost_type(::Type{AffineNetCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineNetCostModel)(x_width, x_work, x_net) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineNetCostModel)
    return bound_stripe(A, K, oracle_stripe(StepHint(), mdl, A))
end
function bound_stripe(A::SparseMatrixCSC, K, ocl::AbstractOracleCost{<:AffineNetCostModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = ocl(1, n + 1)
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end



struct NetCostDominanceOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::NetCostDominanceOracle) = ocl.mdl

function oracle_stripe(hint::AbstractHint, mdl::AbstractNetCostModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = rownetcount(hint, A; kwargs...)
        end
        return NetCostDominanceOracle(pos, net, mdl)
    end
end

@inline function (cst::NetCostDominanceOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k...)
    end
end



mutable struct NetCostStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j::Ti
    j′::Ti
    q′::Ti
    x_net::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractNetCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return NetCostStepOracle(A, mdl, ones(Ti, m), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::NetCostStepOracle) = ocl.mdl

@propagate_inbounds function (stp::NextJ{NetCostStepOracle{Tv, Ti, Mdl}})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    x_net -= Δ_net[j]
    ocl.j = j
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
end

@propagate_inbounds function (stp::PrevJ{NetCostStepOracle{Tv, Ti, Mdl}})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    x_net += Δ_net[j + 1]
    ocl.j = j
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
end

@propagate_inbounds function (stp::NextJ′{NetCostStepOracle{Tv, Ti, Mdl}})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    idx = A.rowval
    q′ = ocl.q′
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    q = q′
    q′ = pos[j′]
    Δ_net[j′] = q′ - q
    for _q = q:q′ - 1
        i = idx[_q]
        j₀ = hst[i] - 1
        x_net += j₀ < j
        Δ_net[j₀ + 1] -= 1
        hst[i] = j′
    end
    ocl.q′ = q′
    ocl.j′ = j′
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
end

@propagate_inbounds function (stp::NextK{NetCostStepOracle{Tv, Ti, Mdl}})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    x_net = ocl.x_net
    return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
end

@inline function (ocl::NetCostStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    @inbounds begin
        ocl_j = ocl.j
        ocl_j′ = ocl.j′
        q′ = ocl.q′
        x_net = ocl.x_net
        Δ_net = ocl.Δ_net
        A = ocl.A
        pos = A.colptr
        idx = A.rowval
        hst = ocl.hst

        if j′ < ocl_j′
            ocl_j = Ti(1)
            ocl_j′ = Ti(1)
            q′ = Ti(1)
            x_net = Ti(0)
            one!(ocl.hst)
        end
        while ocl_j′ < j′
            q = q′
            q′ = pos[ocl_j′ + 1]
            Δ_net[ocl_j′ + 1] = q′ - q
            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                x_net += j₀ < ocl_j
                Δ_net[j₀ + 1] -= 1
                hst[i] = ocl_j′ + 1
            end
            ocl_j′ += 1
        end
        if j == j′ - 1
            ocl_j = j′ - 1
            x_net = Δ_net[ocl_j + 1]
        elseif j == j′
            ocl_j = j′
            x_net = Ti(0)
        else
            while j < ocl_j
                ocl_j -= 1
                x_net += Δ_net[ocl_j + 1]
            end
            while j > ocl_j
                x_net -= Δ_net[ocl_j + 1]
                ocl_j += 1
            end
        end

        ocl.j = ocl_j
        ocl.j′ = ocl_j′
        ocl.q′ = q′
        ocl.x_net = x_net
        return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
    end
end