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

struct AffineFillNetCostModel{Tv} <: AbstractNetCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

@inline cost_type(::Type{AffineFillNetCostModel{Tv}}) where {Tv} = Tv

(mdl::AffineFillNetCostModel)(x_width, x_work, x_net) = mdl.α + x_width * mdl.β_width + (x_width * x_net) * mdl.β_work + x_net * mdl.β_net 

struct NetCostOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::SparseCountedRowNet{Ti}
    mdl::Mdl
end

oracle_model(ocl::NetCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::NetCostOracle{<:Any, <:AffineNetCostModel})
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
        return NetCostOracle(pos, net, mdl)
    end
end

@inline function (cst::NetCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::SplitPartition, mdl::AbstractNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
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
            x_work += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < j
                    x_net += 1
                end
                hst[i] = j
            end
        end
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Φ::DomainPartition, mdl::AbstractNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
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
            x_work += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < s
                    x_net += 1
                end
                hst[i] = s
            end
        end
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Φ::MapPartition, mdl::AbstractNetCostModel)
    return compute_objective(g, A, convert(DomainPartition, Φ), mdl)
end



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

struct SymCostOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    wrk::Vector{Ti}
    net::SparseCountedRowNet{Ti}
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

        net = SparseCountedRowNet{Ti}(n, pos′, SparseCountedArea{Ti}(n + 1, n + 1, N′, pos′, idx′; kwargs...))

        return SymCostOracle(wrk, net, mdl)
    end
end

@inline function (cst::SymCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.wrk[j′] - cst.wrk[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k)
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

struct CommCostOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::SparseCountedRowNet{Ti}
    lcr::SparseCountedLocalRowNet{Ti}
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

function oracle_stripe(mdl::AbstractCommCostModel, A::SparseMatrixCSC, Π; net=nothing, adj_A=nothing, kwargs...)
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

struct LocalCostOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    Π::SplitPartition{Ti}
    pos::Vector{Ti}
    lcc::SparseCountedLocalColNet{Ti}
    mdl::Mdl
end

oracle_model(ocl::LocalCostOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, Π::SplitPartition, ocl::LocalCostOracle{<:Any, <:AffineLocalCostModel})
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

function oracle_stripe(mdl::AbstractLocalCostModel, A::SparseMatrixCSC, Π::SplitPartition; adj_pos=nothing, adj_A=nothing, net=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)

        if adj_A === nothing
            adj_A = adjointpattern(A)
        end

        if adj_pos === nothing
            adj_pos = adj_A.colptr
        end

        lcc = localcolnetcount(A, convert(MapPartition, Π); kwargs...)

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

function compute_objective(g, A, Π, Φ, mdl::AffineLocalCostModel)
    return compute_objective(g, adjointpattern(A), Φ, Π, AffineCommCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end
