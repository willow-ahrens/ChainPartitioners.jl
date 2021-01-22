abstract type AbstractSymCostModel end

@inline (mdl::AbstractSymCostModel)(x_width, x_work, x_net, k) = mdl(x_width, x_work, x_net)

struct AffineSymCostModel{Tv} <: AbstractSymCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
    Δ_work::Tv
end

@inline cost_type(::Type{AffineSymCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineSymCostModel)(x_width, x_work, x_net) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net

struct SymCostOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    wrk::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::SymCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::AbstractOracleCost{<:AffineSymCostModel})
    m, n = size(A)
    @assert m == n
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = ocl(1, n + 1)
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end
function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineSymCostModel)
    @inbounds begin
        m, n = size(A)
        @assert m == n
        N = nnz(A)
        x_work = 0
        for j = 1:n
            x_work += max(A.colptr[j + 1] - A.colptr[j] - mdl.Δ_work, 0)
        end
        c_hi = mdl.α + mdl.β_width * n + mdl.β_work * x_work + mdl.β_net * m
        c_lo = mdl.α + fld(c_hi - mdl.α, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractSymCostModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        @assert m == n
        N = nnz(A)
        wrk = undefs(eltype(A.colptr), n + 1)
        wrk[1] = 1
        for j = 1:n
            wrk[j + 1] = wrk[j] + max(A.colptr[j + 1] - A.colptr[j] - mdl.Δ_work, 0)
        end

        pos = A.colptr
        idx = A.rowval

        #The remaining lines are a more efficient expression of net = rownetcount(A + UniformScaling(true))

        hst = zeros(Ti, m)
        pos′ = undefs(Ti, n + 1)
        idx′ = undefs(Ti, N + n)

        q′ = 1
        for j = 1:n
            pos′[j] = q′
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                idx′[q′] = (n + 1) - hst[i]
                hst[i] = j
                q′ += 1
            end
            if hst[j] < j
                idx′[q′] = (n + 1) - hst[j]
                hst[j] = j
                q′ += 1
            end
        end
        pos′[n + 1] = q′
        N′ = q′ - 1
        resize!(idx′, N′)

        net = SparseCountedRowNet(n, pos′, SparseCountedArea{Ti}(hint, n + 1, n + 1, N′, pos′, idx′; kwargs...))

        return SymCostOracle(wrk, net, mdl)
    end
end

@inline function (cst::SymCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.wrk[j′] - cst.wrk[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k...)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::SplitPartition, mdl::AbstractSymCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    @assert m == n
    hst = zeros(m)
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        x_width = j′ - j
        x_work = 0
        x_net = 0
        for _j = j:(j′ - 1)
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += max(q′ - q - mdl.Δ_work, 0)
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < j
                    x_net += 1
                end
                hst[i] = j
            end
            if hst[_j] < j
                x_net += 1
            end
            hst[_j] = j
        end
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::DomainPartition, mdl::AbstractSymCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    @assert m == n
    hst = zeros(m)
    for k = 1:Φ.K
        s = Φ.spl[k]
        s′ = Φ.spl[k + 1]
        x_width = s′ - s
        x_work = 0
        x_net = 0
        for _s = s:(s′ - 1)
            _j = Φ.prm[_s]
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += max(q′ - q - mdl.Δ_work, 0)
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < s
                    x_net += 1
                end
                hst[i] = s
            end
            if hst[_j] < s
                x_net += 1
            end
            hst[_j] = s
        end
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Φ::MapPartition, mdl::AbstractSymCostModel)
    return compute_objective(g, A, convert(DomainPartition, Φ), mdl)
end