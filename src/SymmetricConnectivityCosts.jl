abstract type AbstractSymmetricConnectivityModel end

@inline (mdl::AbstractSymmetricConnectivityModel)(n_vertices, n_pins, n_nets, k) = mdl(n_vertices, n_pins, n_nets)

struct AffineSymmetricConnectivityModel{Tv} <: AbstractSymmetricConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
    Δ_pins::Tv
end

function AffineSymmetricConnectivityModel(; α = false, β_vertex = false, β_pin = false, β_net = false, Δ_pins = false)
    AffineSymmetricConnectivityModel(promote(α, β_vertex, β_pin, β_net, Δ_pins)...)
end

@inline cost_type(::Type{AffineSymmetricConnectivityModel{Tv}}) where {Tv} = Tv

(mdl::AffineSymmetricConnectivityModel)(n_vertices, n_pins, n_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net

function bound_stripe(A::SparseMatrixCSC, K, ocl::AbstractOracleCost{<:AffineSymmetricConnectivityModel})
    m, n = size(A)
    @assert m == n
    N = nnz(A)
    mdl = oracle_model(ocl)
    c_hi = ocl(1, n + 1)
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineSymmetricConnectivityModel)
    @inbounds begin
        m, n = size(A)
        @assert m == n
        N = nnz(A)
        n_pins = 0
        for j = 1:n
            n_pins += max(A.colptr[j + 1] - A.colptr[j] - mdl.Δ_pins, 0)
        end
        c_hi = mdl.α + mdl.β_vertex * n + mdl.β_pin * n_pins + mdl.β_net * m
        c_lo = mdl.α + fld(c_hi - mdl.α, K)
        return (c_lo, c_hi)
    end
end

struct SymmetricConnectivityOracle{Ti, Net, Mdl} <: AbstractOracleCost{Mdl}
    wrk::Vector{Ti}
    net::Net
    mdl::Mdl
end

oracle_model(ocl::SymmetricConnectivityOracle) = ocl.mdl


function oracle_stripe(hint::AbstractHint, mdl::AbstractSymmetricConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; net=nothing, adj_A=nothing, kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        @assert m == n
        N = nnz(A)
        wrk = undefs(eltype(A.colptr), n + 1)
        wrk[1] = 1
        for j = 1:n
            wrk[j + 1] = wrk[j] + max(A.colptr[j + 1] - A.colptr[j] - mdl.Δ_pins, 0)
        end

        pos = A.colptr
        idx = A.rowval

        #The remaining lines are a more efficient expression of net = netcount(A + UniformScaling(true))

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

        net = NetCount(n, pos′, DominanceCount{Ti}(hint, n + 1, n + 1, N′, pos′, idx′; kwargs...))

        return SymmetricConnectivityOracle(wrk, net, mdl)
    end
end

@inline function (cst::SymmetricConnectivityOracle{Ti, Mdl})(j::Ti, j′::Ti, k...) where {Ti, Mdl}
    @inbounds begin
        w = cst.wrk[j′] - cst.wrk[j]
        d = cst.net[j, j′]
        return cst.mdl(j′ - j, w, d, k...)
    end
end

@inline function (stp::Step{Ocl})(_j, _j′, _k...) where {Ti, Mdl, Ocl <: SymmetricConnectivityOracle{Ti, Mdl}}
    @inbounds begin
        cst = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = maptuple(destep, _k...)
        w = cst.wrk[j′] - cst.wrk[j]
        d = Step(cst.net)(_j, _j′)
        return cst.mdl(j′ - j, w, d, k...)
    end
end

#=
mutable struct SymmetricConnectivityStepOracle{Tv, Ti, Mdl} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ_net::Vector{Ti}
    j::Ti
    j′::Ti
    n_pins::Ti
    n_nets::Ti
end

function oracle_stripe(hint::StepHint, mdl::AbstractSymmetricConnectivityModel, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        return SymmetricConnectivityStepOracle(A, mdl, ones(Ti, m), undefs(Ti, n + 1), Ti(1), Ti(1), Ti(0), Ti(0))
    end
end

oracle_model(ocl::SymmetricConnectivityStepOracle) = ocl.mdl

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: SymmetricConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    n_pins = ocl.n_pins
    n_nets = ocl.n_nets
    return ocl.mdl(j′ - j, n_pins, n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Next{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: SymmetricConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    n_pins = ocl.n_pins
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    Δ_pins = ocl.mdl.Δ_pins
    hst = ocl.hst
    n_nets -= Δ_net[j]
    n_pins -= max(pos[j] - pos[j - 1] - Δ_pins, 0)
    ocl.j = j
    ocl.n_pins = n_pins
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, n_pins, n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Prev{Ti}, _j′::Same{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: SymmetricConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    n_pins = ocl.n_pins
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    Δ_pins = ocl.mdl.Δ_pins
    hst = ocl.hst
    n_nets += Δ_net[j + 1]
    n_pins += max(pos[j + 1] - pos[j] - Δ_pins, 0)
    ocl.j = j
    ocl.n_pins = n_pins
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, n_pins, n_nets, k...)
end

@propagate_inbounds function (stp::Step{Ocl})(_j::Same{Ti}, _j′::Next{Ti}, _k...) where {Tv, Ti, Mdl, Ocl <: SymmetricConnectivityStepOracle{Tv, Ti, Mdl}}
    j = destep(_j)
    j′ = destep(_j′)
    k = maptuple(destep, _k...)
    ocl = stp.ocl
    A = ocl.A
    pos = A.colptr
    idx = A.rowval
    n_pins = ocl.n_pins
    n_nets = ocl.n_nets
    Δ_net = ocl.Δ_net
    Δ_pins = ocl.mdl.Δ_pins
    hst = ocl.hst
    q = pos[j′ - 1]
    q′ = pos[j′]
    Δ_net[j′] = q′ - q
    n_pins += max(q′ - q - Δ_pins, 0)
    for _q = q:q′ - 1
        i = idx[_q]
        j₀ = hst[i] - 1
        n_nets += j₀ < j
        Δ_net[j₀ + 1] -= 1
        hst[i] = j′
    end
    j₀ = hst[j′ - 1] - 1
    Δ_net[j′] += j₀ < j′ - 1
    n_nets += j₀ < j
    Δ_net[j₀ + 1] -= j₀ < j′ - 1
    hst[j′ - 1] = j′
    ocl.j′ = j′
    ocl.n_pins = n_pins
    ocl.n_nets = n_nets
    return ocl.mdl(j′ - j, n_pins, n_nets, k...)
end

@inline function (ocl::SymmetricConnectivityStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    begin
        ocl_j = ocl.j
        ocl_j′ = ocl.j′
        n_pins = ocl.n_pins
        n_nets = ocl.n_nets
        Δ_net = ocl.Δ_net
        Δ_pins = ocl.mdl.Δ_pins
        A = ocl.A
        pos = A.colptr
        idx = A.rowval
        hst = ocl.hst

        if j′ < ocl_j′
            ocl_j = Ti(1)
            ocl_j′ = Ti(1)
            n_pins = Ti(0)
            n_nets = Ti(0)
            one!(ocl.hst)
        end
        while ocl_j′ < j′
            q = pos[ocl_j′]
            q′ = pos[ocl_j′ + 1]
            Δ_net[ocl_j′ + 1] = q′ - q
            n_pins += max(q′ - q - Δ_pins, 0)
            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                n_nets += j₀ < ocl_j
                Δ_net[j₀ + 1] -= 1
                hst[i] = ocl_j′ + 1
            end
            j₀ = hst[ocl_j′] - 1
            Δ_net[ocl_j′ + 1] += j₀ < ocl_j′
            n_nets += j₀ < ocl_j
            Δ_net[j₀ + 1] -= j₀ < ocl_j′
            hst[ocl_j′] = ocl_j′ + 1
            ocl_j′ += 1
        end
        if j == j′ - 1
            ocl_j = j′ - 1
            q = pos[ocl_j]
            q′ = pos[ocl_j + 1]
            n_pins = max(q′ - q - Δ_pins, 0)
            n_nets = Δ_net[ocl_j + 1]
        elseif j == j′
            ocl_j = j′
            n_pins = Ti(0)
            n_nets = Ti(0)
        else
            while j < ocl_j
                ocl_j -= 1
                n_pins += max(pos[ocl_j + 1] - pos[ocl_j] - Δ_pins, 0)
                n_nets += Δ_net[ocl_j + 1]
            end
            while j > ocl_j
                n_pins -= max(pos[ocl_j + 1] - pos[ocl_j] - Δ_pins, 0)
                n_nets -= Δ_net[ocl_j + 1]
                ocl_j += 1
            end
        end

        ocl.j = ocl_j
        ocl.j′ = ocl_j′
        ocl.n_pins = n_pins
        ocl.n_nets = n_nets
        return ocl.mdl(j′ - j, n_pins, n_nets, k...)
    end
end
=#