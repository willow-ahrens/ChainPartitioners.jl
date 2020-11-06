abstract type AbstractNetCostModel end

struct AffineNetCostModel{Tv} <: AbstractNetCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

(mdl::AffineNetCostModel)(x_width, x_work, x_net, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net 

struct NetCostOracle{Ti, Mdl} <: AbstractCostOracle
    pos::Vector{Ti}
    net::SparseCountedRowNet{Ti}
    mdl::Mdl
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::NetCostOracle{<:Any, <:AffineNetCostModel})
    m, n = size(A)
    N = nnz(A)
    c_hi = mdl.mdl.α + mdl.mdl.β_width * n + mdl.mdl.β_work * N + mdl.mdl.β_net * mdl.net[1, end]
    c_lo = mdl.mdl.α + fld(c_hi - mdl.mdl.α, K)
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

function oracle_stripe(mdl::AbstractNetCostModel, A::SparseMatrixCSC, K; net=nothing, adj_A=nothing, kwargs...)
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

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::SplitPartition, mdl::AbstractNetCostModel)
    cst = -Inf
    m, n = size(A)
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::DomainPartition, mdl::AbstractNetCostModel)
    cst = -Inf
    m, n = size(A)
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::MapPartition, mdl::AbstractNetCostModel)
    return bottleneck_stripe(A, K, convert(DomainPartition, Φ), mdl)
end



abstract type AbstractSymCostModel end

struct AffineSymCostModel{Tv} <: AbstractSymCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
    Δ_work::Tv
end

(mdl::AffineSymCostModel)(x_width, x_work, x_net, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net

struct SymCostOracle{Ti, Mdl} <: AbstractCostOracle
    wrk::Vector{Ti}
    net::SparseCountedRowNet{Ti}
    mdl::Mdl
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::SymCostOracle{<:Any, <:AffineSymCostModel})
    m, n = size(A)
    @assert m == n
    N = nnz(A)
    c_hi = mdl.mdl.α + mdl.mdl.β_width * n + mdl.mdl.β_work * (mdl.wrk[end] - mdl.wrk[1]) + mdl.mdl.β_net * m
    c_lo = mdl.mdl.α + fld(c_hi - mdl.mdl.α, K)
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

function oracle_stripe(mdl::AbstractSymCostModel, A::SparseMatrixCSC{Tv, Ti}, K; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
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

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::SplitPartition, mdl::AbstractSymCostModel)
    cst = -Inf
    m, n = size(A)
    @assert m == n
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::DomainPartition, mdl::AbstractSymCostModel)
    cst = -Inf
    m, n = size(A)
    @assert m == n
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function bottleneck_stripe(A::SparseMatrixCSC, K, Φ::MapPartition, mdl::AbstractSymCostModel)
    return bottleneck_stripe(A, K, convert(DomainPartition, Φ), mdl)
end



abstract type AbstractCommCostModel end

struct AffineCommCostModel{Tv} <: AbstractCommCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

(mdl::AffineCommCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

struct CommCostOracle{Ti, Mdl} <: AbstractCostOracle
    pos::Vector{Ti}
    net::SparseCountedRowNet{Ti}
    lcr::SparseCountedLocalRowNet{Ti}
    mdl::Mdl
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::CommCostOracle{<:Any, <:AffineCommCostModel})
    m, n = size(A)
    N = nnz(A)
    c_hi = mdl.mdl.α + mdl.mdl.β_width * n + mdl.mdl.β_work * N + mdl.mdl.β_comm * mdl.net[1, end]
    c_lo = mdl.mdl.α + fld(mdl.mdl.β_width * n + mdl.mdl.β_work * N, K)
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

function oracle_stripe(mdl::AbstractCommCostModel, A::SparseMatrixCSC, K, Π; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = rownetcount(A; kwargs...)
        end
        lcr = localrownetcount(A, K, convert(MapPartition, Π); kwargs...)
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

bottleneck_plaid(A::SparseMatrixCSC, K, Π, Φ, mdl::AbstractCommCostModel) =
    bottleneck_plaid(A, K, convert(MapPartition, Π), Φ, mdl)

function bottleneck_plaid(A::SparseMatrixCSC, K, Π::MapPartition, Φ::SplitPartition, mdl::AbstractCommCostModel)
    cst = -Inf
    m, n = size(A)
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_local, x_comm, k))
    end
    return cst
end

function bottleneck_plaid(A::SparseMatrixCSC, K, Π::MapPartition, Φ::DomainPartition, mdl::AbstractCommCostModel)
    cst = -Inf
    m, n = size(A)
    hst = zeros(m)
    for k = 1:K
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
        cst = max(cst, mdl(x_width, x_work, x_local, x_comm, k))
    end
    return cst
end

function bottleneck_plaid(A::SparseMatrixCSC, K, Π::MapPartition, Φ::MapPartition, mdl::AbstractCommCostModel)
    return bottleneck_plaid(A, K, Π, convert(DomainPartition, Φ), mdl)
end



abstract type AbstractLocalCostModel end

struct AffineLocalCostModel{Tv} <: AbstractLocalCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_local::Tv
    β_comm::Tv
end

(mdl::AffineLocalCostModel)(x_width, x_work, x_local, x_comm, k) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_local * mdl.β_local + x_comm * mdl.β_comm

function bound_stripe(A::SparseMatrixCSC, K, Π, mdl::AffineLocalCostModel)
    adj_A = adjointpattern(A)
    c_hi = bottleneck_stripe(adj_A, K, Π, AffineNetCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_comm))
    c_lo = bottleneck_stripe(adj_A, K, Π, AffineWorkCostModel(mdl.α, mdl.β_width, mdl.β_work))
    return (c_lo, c_hi)
end

struct LocalCostOracle{Ti, Mdl} <: AbstractCostOracle
    Π::SplitPartition{Ti}
    pos::Vector{Ti}
    lcc::SparseCountedLocalColNet{Ti}
    mdl::Mdl
end

function bound_stripe(A::SparseMatrixCSC, K, Π::SplitPartition, mdl::LocalCostOracle{<:Any, <:AffineLocalCostModel})
    @inbounds begin
        c_lo = 0
        c_hi = 0
        for k = 1:K
            x_width = Π.spl[k + 1] - Π.spl[k]
            x_work = mdl.pos[Π.spl[k + 1]] - mdl.pos[Π.spl[k]]
            x_comm = mdl.lcc.Πos[k + 1] - mdl.lcc.Πos[k]
            c_lo = max(c_lo, mdl.mdl.α + x_width * mdl.mdl.β_width + x_work * mdl.mdl.β_work)
            c_hi = max(c_hi, mdl.mdl.α + x_width * mdl.mdl.β_width + x_work * mdl.mdl.β_work + x_comm * mdl.mdl.β_comm)
        end
        return (c_lo, c_hi)
    end
end

function oracle_stripe(mdl::AbstractLocalCostModel, A::SparseMatrixCSC, K, Π::SplitPartition; adj_pos=nothing, adj_A=nothing, net=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)

        if adj_A === nothing
            adj_A = adjointpattern(A)
        end

        if adj_pos === nothing
            adj_pos = adj_A.colptr
        end

        lcc = localcolnetcount(A, K, convert(MapPartition, Π); kwargs...)

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

#Just a bit of a hack for now
function bottleneck_plaid(A, K, Π, Φ, mdl::AffineLocalCostModel)
    return bottleneck_plaid(adjointpattern(A), K, Φ, Π, AffineCommCostModel(mdl.α, mdl.β_width, mdl.β_work, mdl.β_local, mdl.β_comm))
end