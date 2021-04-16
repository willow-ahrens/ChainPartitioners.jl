abstract type AbstractConnectivityModel end

@inline (mdl::AbstractConnectivityModel)(n_vertices, n_pins, n_nets, k) = mdl(n_vertices, n_pins, n_nets)



struct AffineConnectivityModel{Tv} <: AbstractConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
end

@inline cost_type(::Type{AffineConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffineConnectivityModel)(n_vertices, n_pins, n_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net 

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineConnectivityModel)
    return bound_stripe(A, K, oracle_stripe(StepHint(), mdl, A))
end
function bound_stripe(A::SparseMatrixCSC, K, ocl::AbstractOracleCost{<:AffineConnectivityModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = ocl(1, n + 1)
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end



struct ConnectivityOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::ConnectivityOracle) = ocl.mdl

function oracle_stripe(hint::AbstractHint, mdl::AbstractConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if net === nothing
            net = netcount(hint, A; kwargs...)
        end
        return ConnectivityOracle(pos, net, mdl)
    end
end

@inline function (cst::ConnectivityOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k...)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k...) where {Ti, Mdl, Ocl <: ConnectivityOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = maptuple(destep, _k...)
        w = cst.pos[j′] - cst.pos[j]
        d = Step(cst.net)(_j, _j′)
        return cst.mdl(j′ - j, w, d, k...)
    end
end


#=
mutable struct ConnectivityStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j::Ti
    j′::Ti
    q′::Ti
    n_nets::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return ConnectivityStepOracle(A, mdl, ones(Ti, m), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(1), Ti(0))
    end
end

oracle_model(ocl::ConnectivityStepOracle) = ocl.mdl

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: ConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    q′ = ocl.q′
    n_nets = ocl.n_nets
    return ocl.mdl(j′ - j, q′ - pos[j], n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Next{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: ConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
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

@propagate_inbounds function (stp::Step{Ocl})(_j::Prev{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: ConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
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
    k = maptuple(destep, _k...)
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


#=
function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{F}, args...; kwargs...) where {F<:AbstractConnectivityModel, Tv, Ti}
    return pack_stripe(A, DynamicTotalChunker(ConstrainedCost(method.f, FeasibleCost(), Feasible())), args..., kwargs...)
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{<:ConstrainedCost{F}}, args...; n_nets = nothing, kwargs...) where {F<:AbstractConnectivityModel, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        f = method.f.f
        w = oracle_stripe(StepHint(), method.f.w, A, args...) #TODO reverse outer loop so step oracles can have stationary j′
        w_max = method.f.w_max

        Δ_net = zeros(Int, n + 1) # Δ_net is the number of additional distinct entries we see as our part size grows.
        hst = fill(n + 1, m) # hst is the last time we saw some nonzero
        cst = Vector{cost_type(f)}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        spl = Vector{Int}(undef, n + 1)
        Δ_net[n + 1] = 0
        cst[n + 1] = zero(cost_type(f))
        for j = n:-1:1
            d = A_pos[j + 1] - A_pos[j] # The number of distinct nonzero blocks in each candidate part
            Δ_net[j] = d
            for q = A_pos[j] : A_pos[j + 1] - 1
                i = A_idx[q]
                j′ = hst[i]
                Δ_net[j′] -= 1
                hst[i] = j
            end
            best_c = cst[j + 1] + f(1, d, d)
            best_d = d
            best_j′ = j + 1
            for j′ = j + 2 : n + 1
                if w(j, j′) > w_max
                    break
                end
                d += Δ_net[j′ - 1]
                c = cst[j′] + f(j′ - j, A_pos[j′] - A_pos[j], d) 
                if c < best_c
                    best_c = c
                    best_d = d
                    best_j′ = j′
                end
            end
            cst[j] = best_c
            spl[j] = best_j′
        end

        K = 0
        j = 1
        while j != n + 1
            j′ = spl[j]
            K += 1
            spl[K] = j
            j = j′
        end
        spl[K + 1] = j
        resize!(spl, K + 1)
        return SplitPartition{Ti}(K, spl)
    end
end
=#