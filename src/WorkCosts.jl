abstract type AbstractWorkCostModel end

@inline (mdl::AbstractWorkCostModel)(x_width, x_work, k) = mdl(x_width, x_work)

struct AffineWorkCostModel{Tv} <: AbstractWorkCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
end

@inline cost_type(::Type{AffineWorkCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineWorkCostModel)(x_width, x_work) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work

struct WorkCostOracle{Ti, Mdl <: AbstractWorkCostModel} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    mdl::Mdl
end

oracle_model(ocl::WorkCostOracle) = ocl.mdl

function oracle_stripe(mdl::AbstractWorkCostModel, A::SparseMatrixCSC; kwargs...)
    return WorkCostOracle(A.colptr, mdl)
end

@inline function (cst::WorkCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        return cst.mdl(j′ - j, w, k...)
    end
end

bound_stripe(A::SparseMatrixCSC, K, ocl::WorkCostOracle{<:Any, <:AffineWorkCostModel}) = 
    bound_stripe(A, K, oracle_model(ocl))
function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineWorkCostModel)
    m, n = size(A)
    N = nnz(A)
    c_lo = mdl.α + fld(mdl.β_width * n + mdl.β_work * N, K)
    if mdl.β_width ≥ 0 && mdl.β_work ≥ 0
        c_hi = mdl.α + mdl.β_width * n + mdl.β_work * N
    elseif mdl.β_width ≤ 0 && mdl.β_work ≤ 0
        c_hi = mdl.α 
    else
        @assert false
    end
    return (c_lo, c_hi)
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::SplitPartition, mdl::AbstractWorkCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    for k = 1:Π.K
        j = Π.spl[k]
        j′ = Π.spl[k + 1]
        cst = g(cst, mdl(j′ - j, A.colptr[j′] - A.colptr[j], k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::DomainPartition, mdl::AbstractWorkCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    for k = 1:Π.K
        s = Π.spl[k]
        s′ = Π.spl[k + 1]
        x_width = s′ - s
        x_work = 0
        for _s = s : s′ - 1
            j = Π.prm[_s]
            x_work += A.colptr[j + 1] - A.colptr[j]
        end
        cst = g(cst, mdl(x_width, x_work, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, mdl::AbstractWorkCostModel)
    return compute_objective(g, A, convert(DomainPartition, Π), mdl)
end