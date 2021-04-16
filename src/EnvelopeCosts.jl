abstract type AbstractEnvConnectivityModel end

@inline (mdl::AbstractEnvConnectivityModel)(n_vertices, n_pins, n_nets, k) = mdl(n_vertices, n_pins, n_nets)

struct AffineEnvConnectivityModel{Tv} <: AbstractEnvConnectivityModel
    α::Tv
    β_vertex::Tv
    β_pin::Tv
    β_net::Tv
end

@inline cost_type(::Type{AffineEnvConnectivityModel{Tv}}) where {Tv} = Tv

AffineEnvConnectivityModel(α, β_vertex, β_pin, β_net, k) = AffineEnvConnectivityModel(α, β_vertex, β_pin, β_net, k)

(mdl::AffineEnvConnectivityModel)(n_vertices, n_pins, n_nets) = mdl.α + n_vertices * mdl.β_vertex + n_pins * mdl.β_pin + n_nets * mdl.β_net

struct EnvConnectivityOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    env::EnvelopeMatrix{Ti}
    mdl::Mdl
end

oracle_model(ocl::EnvConnectivityOracle) = ocl.mdl

function upperbound_stripe(A::SparseMatrixCSC, K, ocl::EnvConnectivityOracle{<:Any, <:AffineEnvConnectivityModel})
    m, n = size(A)
    N = nnz(A)
    mdl = oracle_model(ocl)
    (env_lo, env_hi) = ocl.env[1, end]
    return mdl.α + mdl.β_vertex * n + mdl.β_pin * N + mdl.β_net * (env_hi - env_lo)
end
function upperbound_stripe(A::SparseMatrixCSC, K, mdl::AffineEnvConnectivityModel)
    m, n = size(A)
    N = nnz(A)
    (env_lo, env_hi) = extrema(A.rowval)
    return mdl.α + mdl.β_vertex * n + mdl.β_pin * N + mdl.β_net * (env_hi - env_lo)
end

function lowerbound_stripe(A::SparseMatrixCSC, K, mdl::EnvConnectivityOracle{<:Any, <:AffineEnvConnectivityModel})
    return fld(upperbound_stripe(A, K, mdl), K) #fld is not strictly correct here
end
function lowerbound_stripe(A::SparseMatrixCSC, K, mdl::AffineEnvConnectivityModel)
    return fld(upperbound_stripe(A, K, mdl), K)
end

function oracle_stripe(hint::AbstractHint, mdl::AbstractEnvConnectivityModel, A::SparseMatrixCSC; env=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if env === nothing
            env = rowenvelope(A) #TODO hint
        end
        return EnvConnectivityOracle(pos, env, mdl)
    end
end

@inline function (cst::EnvConnectivityOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d_lo, d_hi = cst.env[j, j′]
        d = max(d_hi - d_lo, 0)
        return cst.mdl(j′ - j, w, d, k)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::SplitPartition, mdl::AbstractEnvConnectivityModel) where {G}
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

function compute_objective(g::G, A::SparseMatrixCSC, K, Π::DomainPartition, mdl::AbstractEnvConnectivityModel) where {G}
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

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, mdl::AbstractEnvConnectivityModel)
    return compute_objective(g, A, convert(DomainPartition, Π), mdl)
end