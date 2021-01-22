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

struct NetCostDominanceOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::NetCostDominanceOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::NetCostDominanceOracle{<:Any, <:AffineNetCostModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = mdl.α + mdl.β_width * n + mdl.β_work * N + mdl.β_net * ocl.net[1, end]
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineNetCostModel)
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        hst = falses(m)
        x_net = 0
        for j = 1:n
            for q = A.colptr[j]:A.colptr[j+1]-1
                i = A.rowval[q]
                if !hst[i]
                    x_net += 1
                end
                hst[i] = true
            end
        end
        c_hi = mdl.α + mdl.β_width * n + mdl.β_work * N + mdl.β_net * x_net
        c_lo = mdl.α + fld(c_hi - mdl.α, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(mdl::AbstractNetCostModel, A::SparseMatrixCSC; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = rownetcount(A; kwargs...)
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

#=
mutable struct StepOracle{Mdl, Ocl} <: AbstractOracleCost{Mdl}
    ocl::Ocl
    StepOracle{Mdl, Ocl}(ocl::Ocl) where {Mdl, Ocl<:AbstractOracleCost{Mdl}}
        return new{Mdl, Ocl}(ocl)
    end
end
=#

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

function step_oracle_stripe(mdl::AbstractNetCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
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

function compute_objective(g::G, A::SparseMatrixCSC, Φ::SplitPartition, mdl::AbstractNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    ocl = step_oracle_stripe(mdl, A)
    for k = 1:Φ.K
        cst = g(cst, ocl(Φ.spl[k], Φ.spl[k + 1]))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::DomainPartition, mdl::AbstractNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    ocl = step_oracle_stripe(mdl, A[:, Φ.prm])
    for k = 1:Φ.K
        cst = g(cst, ocl(Φ.spl[k], Φ.spl[k + 1]))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Φ::MapPartition, mdl::AbstractNetCostModel)
    return compute_objective(g, A, convert(DomainPartition, Φ), mdl)
end