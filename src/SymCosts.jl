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

struct SymCostOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    wrk::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::SymCostOracle) = ocl.mdl


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



mutable struct SymCostStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j::Ti
    j′::Ti
    x_work::Ti
    x_net::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractSymCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return SymCostStepOracle(A, mdl, ones(Ti, m), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(0), Ti(0))
    end
end

oracle_model(ocl::SymCostStepOracle) = ocl.mdl

@propagate_inbounds function (stp::Step{SymCostStepOracle{Tv, Ti, Mdl}})(_j::Same{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    x_work = ocl.x_work
    x_net = ocl.x_net
    return ocl.mdl(j′ - j, x_work, x_net, k...)
end

@propagate_inbounds function (stp::Step{SymCostStepOracle{Tv, Ti, Mdl}})(_j::Next{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    x_work = ocl.x_work
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    Δ_work = ocl.mdl.Δ_work
    hst = ocl.hst
    x_net -= Δ_net[j]
    x_work -= max(pos[j] - pos[j - 1] - Δ_work, 0)
    ocl.j = j
    ocl.x_work = x_work
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, x_work, x_net, k...)
end

@propagate_inbounds function (stp::Step{SymCostStepOracle{Tv, Ti, Mdl}})(_j::Prev{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    x_work = ocl.x_work
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    Δ_work = ocl.mdl.Δ_work
    hst = ocl.hst
    x_net += Δ_net[j + 1]
    x_work += max(pos[j + 1] - pos[j] - Δ_work, 0)
    ocl.j = j
    ocl.x_work = x_work
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, x_work, x_net, k...)
end

@propagate_inbounds function (stp::Step{SymCostStepOracle{Tv, Ti, Mdl}})(_j::Same{Ti}, _j′::Next{Ti}, _k...) where {Tv, Ti, Mdl}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    idx = A.rowval
    x_work = ocl.x_work
    x_net = ocl.x_net
    Δ_net = ocl.Δ_net
    Δ_work = ocl.mdl.Δ_work
    hst = ocl.hst
    q = pos[j′ - 1]
    q′ = pos[j′]
    Δ_net[j′] = q′ - q
    x_work += max(q′ - q - Δ_work, 0)
    for _q = q:q′ - 1
        i = idx[_q]
        j₀ = hst[i] - 1
        x_net += j₀ < j
        Δ_net[j₀ + 1] -= 1
        hst[i] = j′
    end
    j₀ = hst[j′ - 1] - 1
    Δ_net[j′] += j₀ < j′ - 1
    x_net += j₀ < j
    Δ_net[j₀ + 1] -= j₀ < j′ - 1
    hst[j′ - 1] = j′
    ocl.j′ = j′
    ocl.x_work = x_work
    ocl.x_net = x_net
    return ocl.mdl(j′ - j, x_work, x_net, k...)
end

@inline function (ocl::SymCostStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    begin
        ocl_j = ocl.j
        ocl_j′ = ocl.j′
        x_work = ocl.x_work
        x_net = ocl.x_net
        Δ_net = ocl.Δ_net
        Δ_work = ocl.mdl.Δ_work
        A = ocl.A
        pos = A.colptr
        idx = A.rowval
        hst = ocl.hst

        if j′ < ocl_j′
            ocl_j = Ti(1)
            ocl_j′ = Ti(1)
            x_work = Ti(0)
            x_net = Ti(0)
            one!(ocl.hst)
        end
        while ocl_j′ < j′
            q = pos[ocl_j′]
            q′ = pos[ocl_j′ + 1]
            Δ_net[ocl_j′ + 1] = q′ - q
            x_work += max(q′ - q - Δ_work, 0)
            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                x_net += j₀ < ocl_j
                Δ_net[j₀ + 1] -= 1
                hst[i] = ocl_j′ + 1
            end
            j₀ = hst[ocl_j′] - 1
            Δ_net[ocl_j′ + 1] += j₀ < ocl_j′
            x_net += j₀ < ocl_j
            Δ_net[j₀ + 1] -= j₀ < ocl_j′
            hst[ocl_j′] = ocl_j′ + 1
            ocl_j′ += 1
        end
        if j == j′ - 1
            ocl_j = j′ - 1
            q = pos[ocl_j]
            q′ = pos[ocl_j + 1]
            x_work = max(q′ - q - Δ_work, 0)
            x_net = Δ_net[ocl_j + 1]
        elseif j == j′
            ocl_j = j′
            x_work = Ti(0)
            x_net = Ti(0)
        else
            while j < ocl_j
                ocl_j -= 1
                x_work += max(pos[ocl_j + 1] - pos[ocl_j] - Δ_work, 0)
                x_net += Δ_net[ocl_j + 1]
            end
            while j > ocl_j
                x_work -= max(pos[ocl_j + 1] - pos[ocl_j] - Δ_work, 0)
                x_net -= Δ_net[ocl_j + 1]
                ocl_j += 1
            end
        end

        ocl.j = ocl_j
        ocl.j′ = ocl_j′
        ocl.x_work = x_work
        ocl.x_net = x_net
        return ocl.mdl(j′ - j, x_work, x_net, k...)
    end
end