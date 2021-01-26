abstract type AbstractLocalCostModel end

@inline (mdl::AbstractLocalCostModel)(x_width, x_work, x_net, x_local, k) = mdl(x_width, x_work, x_net, x_local)

struct AffineLocalCostModel{Tv} <: AbstractLocalCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

@inline cost_type(::Type{AffineLocalCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineLocalCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineLocalCostModel)
    adj_A = adjointpattern(A)
    c_hi = bottleneck_value(adj_A, Π, AffineNetCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_comm))
    c_lo = bottleneck_value(adj_A, Π, AffineWorkCostModel(mdl.α, mdl.β_width, mdl.β_work))
    return (c_lo, c_hi)
end

struct LocalCostOracle{Ti, Lcc, Mdl} <: AbstractOracleCost{Mdl}
    Π::SplitPartition{Ti}
    pos::Vector{Ti}
    lcc::Lcc
    mdl::Mdl
end

oracle_model(ocl::LocalCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, Π::SplitPartition, ocl::LocalCostOracle{<:Any, <:Any, <:AffineLocalCostModel})
    @inbounds begin
        c_lo = 0
        c_hi = 0
        mdl = oracle_model(ocl)
        for k = 1:K
            x_width = Π.spl[k + 1] - Π.spl[k]
            x_work = ocl.pos[Π.spl[k + 1]] - ocl.pos[Π.spl[k]]
            x_comm = ocl.lcc.Πos[k + 1] - ocl.lcc.Πos[k]
            c_lo = max(c_lo, mdl.α + x_width * mdl.β_width + x_work * mdl.β_work)
            c_hi = max(c_hi, mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_comm * mdl.β_comm)
        end
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractLocalCostModel, A::SparseMatrixCSC, Π::SplitPartition; adj_pos=nothing, adj_A=nothing, net=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)

        if adj_A === nothing
            adj_A = adjointpattern(A)
        end

        if adj_pos === nothing
            adj_pos = adj_A.colptr
        end

        lcc = localcolnetcount(hint, A, convert(MapPartition, Π); kwargs...)

        return LocalCostOracle(Π, adj_pos, lcc, mdl)
    end
end

@inline function (cst::LocalCostOracle{Ti, Mdl})(i::Ti, i′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[cst.Π.spl[k + 1]] - cst.pos[cst.Π.spl[k]]
        d = cst.lcc.Πos[k + 1] - cst.lcc.Πos[k]
        l = cst.lcc[i, i′, k]
        return cst.mdl(cst.Π.spl[k + 1] - cst.Π.spl[k], w, l, d - l, k)
    end
end

function compute_objective(g::G, A, Π, Φ::SplitPartition, mdl::AffineLocalCostModel) where {G}
    return compute_objective(g, adjointpattern(A), Φ, Π, AffineCommCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end
function compute_objective(g::G, A, Π, Φ, mdl::AffineLocalCostModel) where {G}
    return compute_objective(g, adjointpattern(A), Φ, Π, AffineCommCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end