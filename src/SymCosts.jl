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

(mdl::AffineSymCostModel)(x_width, x_work, x_net, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net

struct SymCostOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    wrk::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::SymCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::SymCostOracle{<:Any, <:AffineSymCostModel})
    m, n = size(A)
    @assert m == n
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = mdl.α + mdl.β_width * n + mdl.β_work * (ocl.wrk[end] - ocl.wrk[1]) + mdl.β_net * m
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

function oracle_stripe(mdl::AbstractSymCostModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
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

        net = SparseCountedRowNet(n, pos′, SparseCountedArea{Ti}(n + 1, n + 1, N′, pos′, idx′; kwargs...))

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

mutable struct SymCostStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j₁::Ti
    j′₀::Ti
    x_net::Ti
end

function step_oracle_stripe(mdl::AbstractSymCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        @assert m == n
        return SymCostStepOracle(A, mdl, ones(Ti, m), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::SymCostStepOracle) = ocl.mdl

@inline function (cst::SymCostStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    @inbounds begin
        if j′ < cst.j′₀
            cst.j = 1
            cst.j′ = 1
            cst.x_net = 0
            one!(cst.hst)
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
            Δ_net[j′₀ + 1] = 1 + q₁ - q₀
            for q = q₀:q₁
                i = A.rowval[q]
                x_net += hst[i] - 1 < j₁
                Δ_net[hst[i]] -= 1
                hst[i] = j′₀ + 1
            end
            if hst[j′₀] - 1 < j′₀
                x_net += 1
                Δ_net[hst[j′₀]] -= 1
            end
            j′₀ += 1
        end
        while j < j₁
            j₁ -= 1
            x_net += Δ_net[j₁ + 1]
        end
        while j > j₁
            x_net -= Δ_net[j₁ + 1]
            j₁ += 1
        end

        cst.j₁ = j₁
        cst.j′₀ = j′₀
        cst.x_net = x_net
        return cst.mdl(j′ - j, A.colptr[j′] - A.colptr[j], x_net, k...)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::SplitPartition, mdl::AbstractSymCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    ocl = step_oracle_stripe(mdl, A)
    for k = 1:Φ.K
        cst = g(cst, ocl(Φ.spl[k], Φ.spl[k + 1]))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::DomainPartition, mdl::AbstractSymCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    ocl = step_oracle_stripe(mdl, A[:, Φ.prm])
    for k = 1:Φ.K
        cst = g(cst, ocl(Φ.spl[k], Φ.spl[k + 1]))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Φ::MapPartition, mdl::AbstractSymCostModel)
    return compute_objective(g, A, convert(DomainPartition, Φ), mdl)
end