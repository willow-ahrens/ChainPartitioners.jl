abstract type AbstractPrimaryConnectivityModel end

@inline (mdl::AbstractPrimaryConnectivityModel)(n_vertices, n_pins, n_nets, n_local_nets, k) = mdl(n_vertices, n_pins, n_nets, n_local_nets)

struct AffinePrimaryConnectivityModel{Tv} <: AbstractPrimaryConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_local_net::Tv
    β_remote_net::Tv
end

function AffinePrimaryConnectivityModel(; α = false, β_vertex = false, β_pin = false, β_local_net = false, β_remote_net = false)
    AffinePrimaryConnectivityModel(promote(α, β_vertex, β_pin, β_local_net, β_remote_net)...)
end

@inline cost_type(::Type{AffinePrimaryConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffinePrimaryConnectivityModel)(n_vertices, n_pins, n_local_nets, n_remote_nets, k) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_local_nets * mdl.β_local_net + n_remote_nets * mdl.β_remote_net

struct PrimaryConnectivityOracle{Ti, Net, Lcr, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    lcr::Lcr
    mdl::Mdl
end

oracle_model(ocl::PrimaryConnectivityOracle) = ocl.mdl

function bound_stripe(A::SparseMatrixCSC, K, ocl::PrimaryConnectivityOracle{<:Any, <:Any, <:Any, <:AffinePrimaryConnectivityModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = mdl.α + mdl.β_vertex * n + mdl.β_pin * N + mdl.β_remote_net * ocl.net[1, end]
    c_lo = mdl.α + fld(mdl.β_vertex * n + mdl.β_pin * N, K)
    return (c_lo, c_hi)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffinePrimaryConnectivityModel)
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        hst = falses(m)
        n_remote_nets = 0
        for j = 1:n
            for q = A.colptr[j]:A.colptr[j+1]-1
                i = A.rowval[q]
                if !hst[i]
                    n_remote_nets += 1
                end
                hst[i] = true
            end
        end
        c_hi = mdl.α + mdl.β_vertex * n + mdl.β_pin * N + mdl.β_remote_net * n_remote_nets
        c_lo = mdl.α + fld(mdl.β_vertex * n + mdl.β_pin * N, K)
        return (c_lo, c_hi)
    end
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractPrimaryConnectivityModel, A::SparseMatrixCSC, Π; net=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = netcount(hint, A; kwargs...)
        end
        args, Ap = partwise(A, convert(MapPartition, Π))
        lcr = partwisecount!(hint, args..., netcount(hint, Ap, kwargs...); kwargs...)
        return PrimaryConnectivityOracle(pos, net, lcr, mdl)
    end
end

@inline function (cst::PrimaryConnectivityOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net(j, j′)
        l = cst.lcr(j, j′, k)
        return cst.mdl(j′ - j, w, l, d - l, k)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k) where {Ti, Mdl, Ocl <: PrimaryConnectivityOracle{Ti, Mdl}}
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

compute_objective(g::G, A::SparseMatrixCSC, Π, Φ, mdl::AbstractPrimaryConnectivityModel) where {G} =
    compute_objective(g, A, convert(MapPartition, Π), Φ, mdl)

compute_objective(g::G, A::SparseMatrixCSC, Π, Φ::SplitPartition, mdl::AbstractPrimaryConnectivityModel) where {G} =
    compute_objective(g, A, convert(MapPartition, Π), Φ, mdl)

function compute_objective(g::G, A::SparseMatrixCSC, Π::MapPartition, Φ::SplitPartition, mdl::AbstractPrimaryConnectivityModel) where {G}
    @assert Φ.K == Π.K
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        n_vertices = j′ - j
        n_pins = 0
        n_local_nets = 0
        n_remote_nets = 0
        for _j = j:(j′ - 1)
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            n_pins += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < j
                    if Π.asg[i] == k
                        n_local_nets += 1
                    else
                        n_remote_nets += 1
                    end
                end
                hst[i] = j
            end
        end
        cst = g(cst, mdl(n_vertices, n_pins, n_local_nets, n_remote_nets, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::MapPartition, Φ::DomainPartition, mdl::AbstractPrimaryConnectivityModel) where {G}
    @assert Φ.K == Π.K
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        s = Φ.spl[k]
        s′ = Φ.spl[k + 1]
        n_vertices = s′ - s
        n_pins = 0
        n_local_nets = 0
        n_remote_nets = 0
        for _s = s:(s′ - 1)
            _j = Φ.prm[_s]
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            n_pins += q′ - q
            for _q = q : q′ - 1
                i = A.rowval[_q]
                if hst[i] < s
                    if Π.asg[i] == k
                        n_local_nets += 1
                    else
                        n_remote_nets += 1
                    end
                end
                hst[i] = s
            end
        end
        cst = g(cst, mdl(n_vertices, n_pins, n_local_nets, n_remote_nets, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, Φ::MapPartition, mdl::AbstractPrimaryConnectivityModel)
    return compute_objective(g, A, Π, convert(DomainPartition, Φ), mdl)
end



#=
mutable struct PrimaryConnectivityStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
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
    n_nets::Ti
    n_local_nets::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return ConnectivityStepOracle(A, mdl, ones(Ti, m), ones(Ti, K), undefs(Ti, n + 1), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::PrimaryConnectivityStepOracle) = ocl.mdl

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: PrimaryConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    n_nets = ocl.n_nets
    return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Next{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: PrimaryConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    n_nets -= Δ_net[j]
    ocl.j = j
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Prev{Ti}, _j′::Same{Ti}, _k::Same{Ti}) where {Tv, Ti, Mdl, Ocl <: ConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    n_nets += Δ_net[j + 1]
    ocl.j = j
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Next{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: ConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = destep(_k)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    idx = A.rowval
    q′ = ocl.q′
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    hst = ocl.hst
    q = q′
    q′ = pos[j′]
    Δ_net[j′] = q′ - q
    for _q = q:q′ - 1
        i = idx[_q]
        j₀ = hst[i] - 1
        n_nets += j₀ < j
        Δ_net[j₀ + 1] -= 1
        hst[i] = j′
    end
    ocl.q′ = q′
    ocl.j′ = j′
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
end

@inline function (ocl::ConnectivityStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    @inbounds begin
        ocl_j = ocl.j
        ocl_j′ = ocl.j′
        q′ = ocl.q′
        n_nets = ocl.n_nets
        Δ_net = ocl.Δ_net
        A = ocl.A
        pos = A.colptr
        idx = A.rowval
        hst = ocl.hst

        if j′ < ocl_j′
            ocl_j = Ti(1)
            ocl_j′ = Ti(1)
            q′ = Ti(1)
            n_nets = Ti(0)
            one!(ocl.hst)
        end
        while ocl_j′ < j′
            q = q′
            q′ = pos[ocl_j′ + 1]
            Δ_net[ocl_j′ + 1] = q′ - q
            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                n_nets += j₀ < ocl_j
                Δ_net[j₀ + 1] -= 1
                hst[i] = ocl_j′ + 1
            end
            ocl_j′ += 1
        end
        if j == j′ - 1
            ocl_j = j′ - 1
            n_nets = Δ_net[ocl_j + 1]
        elseif j == j′
            ocl_j = j′
            n_nets = Ti(0)
        else
            while j < ocl_j
                ocl_j -= 1
                n_nets += Δ_net[ocl_j + 1]
            end
            while j > ocl_j
                n_nets -= Δ_net[ocl_j + 1]
                ocl_j += 1
            end
        end

        ocl.j = ocl_j
        ocl.j′ = ocl_j′
        ocl.q′ = q′
        ocl.n_nets = n_nets
        return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
    end
end
=#