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

function bound_stripe(A::SparseMatrixCSC, K, ocl::CommCostOracle{<:Any, <:Any, <:Any, <:AffineCommCostModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
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
            net = rownetcount(hint, A; kwargs...)
        end
        lcr = localrownetcount(hint, A, convert(MapPartition, Π); kwargs...)
        return CommCostOracle(pos, net, lcr, mdl)
    end
end

@inline function (cst::CommCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net(j, j′)
        l = cst.lcr(j, j′, k)
        return cst.mdl(j′ - j, w, l, d - l, k)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k) where {Ti, Mdl, Ocl <: CommCostOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        w = cst.pos[j′] - cst.pos[j]
        d = Step(cst.net)(_j, _j′)
        l = Step(cst.lcr)(_j, _j′, _k)
        return cst.mdl(j′ - j, w, l, d - l, k)
    end
end

compute_objective(g::G, A::SparseMatrixCSC, Π, Φ, mdl::AbstractCommCostModel) where {G} =
    compute_objective(g, A, convert(MapPartition, Π), Φ, mdl)

compute_objective(g::G, A::SparseMatrixCSC, Π, Φ::SplitPartition, mdl::AbstractCommCostModel) where {G} =
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



#=
mutable struct CommCostStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    Δ_local::Vector{Ti}
    Π::MapPartition{Ti}
    j::Ti
    j′::Ti
    k::Ti
    q′::Ti
    x_net::Ti
    x_local::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractNetCostModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return NetCostStepOracle(A, mdl, ones(Ti, m), ones(Ti, K), undefs(Ti, n + 1), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::CommCostStepOracle) = ocl.mdl

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: CommCostStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    x_net = ocl.x_net
    return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Next{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: CommCostStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
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

@propagate_inbounds function (stp::Step{Ocl})(_j::Prev{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: NetCostStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
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

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Next{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: NetCostStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
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
=#