abstract type AbstractCommCostModel end

@inline (mdl::AbstractCommCostModel)(x_width, x_work, x_net, x_local, k) = mdl(x_width, x_work, x_net, x_local)

struct AffineCommCostModel{Tv} <: AbstractCommCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

@inline cost_type(::Type{AffineCommCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineCommCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

struct CommCostOracle{Ti, Net, Lcr, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    lcr::Lcr
    mdl::Mdl
end

oracle_model(ocl::CommCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::CommCostOracle{<:Any, <:AffineCommCostModel})
    m, n = size(A)
    N = nnz(A)
    oracle_model(ocl)
    c_hi = mdl.α + mdl.β_width * n + mdl.β_work * N + mdl.β_comm * ocl.net[1, end]
    c_lo = mdl.α + fld(mdl.β_width * n + mdl.β_work * N, K)
    return (c_lo, c_hi)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineCommCostModel)
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        hst = falses(m)
        x_comm = 0
        for j = 1:n
            for q = A.colptr[j]:A.colptr[j+1]-1
                i = A.rowval[q]
                if !hst[i]
                    x_comm += 1
                end
                hst[i] = true
            end
        end
        c_hi = mdl.α + mdl.β_width * n + mdl.β_work * N + mdl.β_comm * x_comm
        c_lo = mdl.α + fld(mdl.β_width * n + mdl.β_work * N, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractCommCostModel, A::SparseMatrixCSC, Π; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = rownetcount(A; kwargs...)
        end
        lcr = localrownetcount(A, convert(MapPartition, Π); kwargs...)
        return CommCostOracle(pos, net, lcr, mdl)
    end
end

@inline function (cst::CommCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net[j, j′]
        l = cst.lcr[j, j′, k]
        return cst.mdl(j′ - j, w, l, d - l, k)
    end
end

compute_objective(g, A::SparseMatrixCSC, Π, Φ, mdl::AbstractCommCostModel) =
    compute_objective(g, A, convert(MapPartition, Π), Φ, mdl)

function compute_objective(g::G, A::SparseMatrixCSC, Π::MapPartition, Φ::SplitPartition, mdl::AbstractCommCostModel) where {G}
    @assert Φ.K == Π.K
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        x_width = j′ - j
        x_work = 0
        x_local = 0
        x_comm = 0
        for _j = j:(j′ - 1)
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < j
                    if Π.asg[i] == k
                        x_local += 1
                    else
                        x_comm += 1
                    end
                end
                hst[i] = j
            end
        end
        cst = g(cst, mdl(x_width, x_work, x_local, x_comm, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::MapPartition, Φ::DomainPartition, mdl::AbstractCommCostModel) where {G}
    @assert Φ.K == Π.K
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        s = Φ.spl[k]
        s′ = Φ.spl[k + 1]
        x_width = s′ - s
        x_work = 0
        x_local = 0
        x_comm = 0
        for _s = s:(s′ - 1)
            _j = Φ.prm[_s]
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < s
                    if Π.asg[i] == k
                        x_local += 1
                    else
                        x_comm += 1
                    end
                end
                hst[i] = s
            end
        end
        cst = g(cst, mdl(x_width, x_work, x_local, x_comm, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, Φ::MapPartition, mdl::AbstractCommCostModel)
    return compute_objective(g, A, Π, convert(DomainPartition, Φ), mdl)
end