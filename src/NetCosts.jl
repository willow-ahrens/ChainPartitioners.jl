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

mutable struct NetCostStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j₁::Ti
    j′₀::Ti
    x_net::Ti
end

function step_oracle_stripe(mdl::AbstractNetCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return NetCostStepOracle(A, mdl, zeros(Ti, m), undefs(Ti, n), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::NetCostStepOracle) = ocl.mdl

@inline function (cst::NetCostStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    if j′ < cst.j′₀
        cst.j = 1
        cst.j′ = 1
        cst.x_net = 0
        zero!(cst.hst)
    end
    A = cst.A
    j₁ = cst.j₁
    j′₀ = cst.j′₀
    x_net = cst.x_net
    Δ_net = cst.Δ_net
    hst = cst.hst
    while j′₀ < j′
        q₀ = A.colptr[j′₀]
        q₁ = A.colptr[j′₀ + 1] - 1
        Δ_net[j′₀] = 1 + q₁ - q₀
        for q = q₀:q₁
            i = A.rowval[q]
            x_net += hst[i] < j₁
            if 0 < hst[i] < j′₀
                Δ_net[hst[i]] -= 1
            end
            hst[i] = j′₀
        end
        j′₀ += 1
    end
    while j < j₁
        j₁ -= 1
        x_net += Δ_net[j₁]
    end
    while j > j₁
        x_net -= Δ_net[j₁]
        j₁ += 1
    end

    cst.j₁ = j₁
    cst.j′₀ = j′₀
    cst.x_net = x_net
    return cst.mdl(j′ - j, A.colptr[j′] - A.colptr[j], x_net, k...)
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