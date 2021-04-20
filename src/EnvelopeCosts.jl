abstract type AbstractEnvelopeModel end

@inline (mdl::AbstractEnvelopeModel)(n_vertices, n_pins, n_nets, k) = mdl(n_vertices, n_pins, n_nets)

struct AffineEnvelopeModel{Tv} <: AbstractEnvelopeModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
end

function AffineEnvelopeModel(; α = false, β_vertex = false, β_pin = false, β_net = false)
    AffineEnvelopeModel(promote(α, β_vertex, β_pin, β_net)...)
end

@inline cost_type(::Type{AffineEnvelopeModel{Tv}}) where {Tv} = Tv

AffineEnvelopeModel(α, β_vertex, β_pin, β_net, k) = AffineEnvelopeModel(α, β_vertex, β_pin, β_net, k)

(mdl::AffineEnvelopeModel)(n_vertices, n_pins, n_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net

struct EnvelopeOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    env::EnvelopeMatrix{Ti}
    mdl::Mdl
end

oracle_model(ocl::EnvelopeOracle) = ocl.mdl


function bound_stripe(A::SparseMatrixCSC, K, ocl::AbstractOracleCost{<:AffineEnvelopeModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    @assert mdl.β_vertex >= 0
    @assert mdl.β_pin >= 0
    @assert mdl.β_net >= 0
    c_hi = ocl(1, n + 1)
    c_lo = mdl.α + fld(c_hi - mdl.α, K)
    return (c_lo, c_hi)
end

function bound_stripe(A::SparseMatrixCSC, K, mdl::AffineEnvelopeModel)
    m, n = size(A)
    N = nnz(A)
    @assert mdl.β_vertex >= 0
    @assert mdl.β_pin >= 0
    @assert mdl.β_net >= 0
    (env_lo, env_hi) = extrema(A.rowval)
    c_hi = mdl.α + mdl.β_vertex * n + mdl.β_pin * N + mdl.β_net * (env_hi - env_lo)
    c_lo = mdl.α + fld(mdl.β_vertex * n + mdl.β_pin * N + mdl.β_net * (env_hi - env_lo), K)
    return (c_lo, c_hi)
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractEnvelopeModel, A::SparseMatrixCSC; env=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if env === nothing
            env = rowenvelope(A) #TODO hint
        end
        return EnvelopeOracle(pos, env, mdl)
    end
end

@inline function (cst::EnvelopeOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d_lo, d_hi = cst.env[j, j′]
        d = max(d_hi - d_lo, 0)
        return cst.mdl(j′ - j, w, d, k)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::SplitPartition, mdl::AbstractEnvelopeModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    for k = 1:Π.K
        j = Π.spl[k]
        j′ = Π.spl[k + 1]
        n_vertices = j′ - j
        n_pins = 0
        x_env_lo = m + 1
        x_env_hi = 0
        for _j = j:(j′ - 1)
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            n_pins += q′ - q
            if q′ > q
                x_env_lo = min(x_env_lo, A.rowval[q])
                x_env_hi = max(x_env_hi, A.rowval[q′ - 1])
            end
        end
        n_nets = max(x_env_hi - x_env_lo, 0)
        cst = g(cst, mdl(n_vertices, n_pins, n_nets, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, K, Π::DomainPartition, mdl::AbstractEnvelopeModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        s = Π.spl[k]
        s′ = Π.spl[k + 1]
        n_vertices = s′ - s
        n_pins = 0
        x_env_lo = m + 1
        x_env_hi = 0
        for _s = s:(s′ - 1)
            _j = Π.prm[_s]
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            n_pins += q′ - q
            if q′ > q
                x_env_lo = min(x_env_lo, A.rowval[q])
                x_env_hi = max(x_env_hi, A.rowval[q′ - 1])
            end
        end
        n_nets = max(x_env_hi - x_env_lo, 0)
        cst = g(cst, mdl(n_vertices, n_pins, n_nets, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, mdl::AbstractEnvelopeModel)
    return compute_objective(g, A, convert(DomainPartition, Π), mdl)
end