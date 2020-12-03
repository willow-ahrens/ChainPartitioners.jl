abstract type AbstractEnvNetCostModel end

@inline (mdl::AbstractEnvNetCostModel)(x_width, x_work, x_net, k) = mdl(x_width, x_work, x_net)

struct AffineEnvNetCostModel{Tv} <: AbstractEnvNetCostModel
    α::Tv
    β_width::Tv
    β_work::Tv
    β_net::Tv
end

@inline cost_type(::Type{AffineEnvNetCostModel{Tv}}) where {Tv} = Tv

AffineEnvNetCostModel(α, β_width, β_work, β_net, k) = AffineEnvNetCostModel(α, β_width, β_work, β_net, k)

(mdl::AffineEnvNetCostModel)(x_width, x_work, x_net) = mdl.α + x_width * mdl.β_width + x_work * mdl.β_work + x_net * mdl.β_net

struct EnvNetCostOracle{Ti, Mdl} <: AbstractOracleCost{Mdl}
    pos::Vector{Ti}
    env::EnvelopeMatrix{Ti}
    mdl::Mdl
end

function upperbound_stripe(A::SparseMatrixCSC, K, mdl::EnvNetCostOracle{<:Any, <:AffineEnvNetCostModel})
    m, n = size(A)
    N = nnz(A)
    (env_lo, env_hi) = mdl.env[1, end]
    return mdl.mdl.α + mdl.mdl.β_width * n + mdl.mdl.β_work * N + mdl.mdl.β_net * max(env_hi - env_lo, 0)
end
function upperbound_stripe(A::SparseMatrixCSC, K, mdl::AffineEnvNetCostModel)
    m, n = size(A)
    N = nnz(A)
    (env_lo, env_hi) = extrema(A.rowval)
    return mdl.α + mdl.β_width * n + mdl.β_work * N + mdl.β_net * (env_hi - env_lo)
end

function lowerbound_stripe(A::SparseMatrixCSC, K, mdl::EnvNetCostOracle{<:Any, <:AffineEnvNetCostModel})
    return fld(upperbound_stripe(A, K, mdl), K) #fld is not strictly correct here
end
function lowerbound_stripe(A::SparseMatrixCSC, K, mdl::AffineEnvNetCostModel)
    return fld(upperbound_stripe(A, K, mdl), K)
end

function oracle_stripe(mdl::AbstractEnvNetCostModel, A::SparseMatrixCSC; env=nothing, adj_A=nothing, kwargs...)
    @inbounds begin
        m, n = size(A)
        pos = A.colptr
        if env === nothing
            env = rowenvelope(A)
        end
        return EnvNetCostOracle(pos, env, mdl)
    end
end

@inline function (cst::EnvNetCostOracle{Ti, Mdl})(j::Ti, j′::Ti, k) where {Ti, Mdl}
    @inbounds begin
        w = cst.pos[j′] - cst.pos[j]
        d_lo, d_hi = cst.env[j, j′]
        d = max(d_hi - d_lo, 0)
        return cst.mdl(j′ - j, w, d, k)
    end
end

function compute_objective(g::G, A::SparseMatrixCSC, Π::SplitPartition, mdl::AbstractEnvNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    for k = 1:Π.K
        j = Π.spl[k]
        j′ = Π.spl[k + 1]
        x_width = j′ - j
        x_work = 0
        x_env_lo = m + 1
        x_env_hi = 0
        for _j = j:(j′ - 1)
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += q′ - q
            if q′ > q
                x_env_lo = min(x_env_lo, A.rowval[q])
                x_env_hi = max(x_env_hi, A.rowval[q′ - 1])
            end
        end
        x_net = max(x_env_hi - x_env_lo, 0)
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g::G, A::SparseMatrixCSC, K, Π::DomainPartition, mdl::AbstractEnvNetCostModel) where {G}
    cst = objective_identity(g, cost_type(mdl))
    m, n = size(A)
    hst = zeros(m)
    for k = 1:Π.K
        s = Π.spl[k]
        s′ = Π.spl[k + 1]
        x_width = s′ - s
        x_work = 0
        x_env_lo = m + 1
        x_env_hi = 0
        for _s = s:(s′ - 1)
            _j = Π.prm[_s]
            q = A.colptr[_j]
            q′ = A.colptr[_j + 1]
            x_work += q′ - q
            if q′ > q
                x_env_lo = min(x_env_lo, A.rowval[q])
                x_env_hi = max(x_env_hi, A.rowval[q′ - 1])
            end
        end
        x_net = max(x_env_hi - x_env_lo, 0)
        cst = g(cst, mdl(x_width, x_work, x_net, k))
    end
    return cst
end

function compute_objective(g, A::SparseMatrixCSC, Π::MapPartition, mdl::AbstractEnvNetCostModel)
    return compute_objective(g, A, convert(DomainPartition, Π), mdl)
end