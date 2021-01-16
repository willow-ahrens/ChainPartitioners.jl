abstract type AbstractOracleCost{Mdl} end

function oracle_stripe(mdl, A, Π; kwargs...)
    return oracle_stripe(mdl, A; kwargs...)
end

function bound_stripe(A, K, Π, ocl::AbstractOracleCost)
    return bound_stripe(A, K, Π, oracle_model(ocl))
end

function bound_stripe(A, K, ocl::AbstractOracleCost)
    return bound_stripe(A, K, oracle_model(ocl))
end

function bound_stripe(A, K, Π, mdl)
    return bound_stripe(A, K, mdl)
end

@inline cost_type(mdl) = cost_type(typeof(mdl))
@inline cost_type(::Type{<:AbstractOracleCost{Mdl}}) where {Mdl} = cost_type(Mdl)
@inline objective_identity(::typeof(+), T) = zero(T)
@inline objective_identity(::typeof(max), T) = typemin(T)

@inline bottleneck_value(A, Π, Φ, mdl) = compute_objective(max, A, Π, Φ, mdl)
@inline bottleneck_value(A, Φ, mdl) = compute_objective(max, A, Φ, mdl)
@inline total_value(A, Π, Φ, mdl) = compute_objective(+, A, Π, Φ, mdl)
@inline total_value(A, Φ, mdl) = compute_objective(+, A, Φ, mdl)

@inline function compute_objective(g, A, Π, Φ, mdl)
    return compute_objective(g, A, Π, Φ, oracle_stripe(mdl, A, Π))
end
function compute_objective(g::G, A, Π, Φ, mdl::AbstractOracleCost) where {G}
    cst = objective_identity(g, cost_type(mdl))
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = g(cst, mdl(j, j′, k))
    end
    return cst
end
@inline function compute_objective(g, A, Φ, mdl)
    return compute_objective(g, A, Φ, oracle_stripe(mdl, A))
end
function compute_objective(g, A, Φ::SplitPartition, mdl::AbstractOracleCost) where {G}
    cst = objective_identity(g, cost_type(mdl))
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = g(cst, mdl(j, j′, k))
    end
    return cst
end



struct ConstrainedCost{F, W, Tw}
    f::F
    w::W
    w_max::Tw
    function ConstrainedCost{F, W, Tw}(f::F, w::W, w_max) where {F, W, Tw}
        @assert Tw == cost_type(W)
        return new{F, W, Tw}(f, w, w_max)
    end
end

ConstrainedCost(f::F, w::W, w_max) where {F, W} = ConstrainedCost{F, W, cost_type(W)}(f, w, w_max)

cost_type(::Type{ConstrainedCost{F, W, Tw}}) where {F, W, Tw} = cost_type(F)

struct ConstrainedCostOracle{F, W, Tc, Tw} <: AbstractOracleCost{ConstrainedCost{F, W, Tw}}
    f::F
    w::W
    f_max::Tc
    w_max::Tw
    function ConstrainedCostOracle{F, W, Tc, Tw}(f::F, w::W, f_max, w_max) where {F, W, Tc, Tw}
        @assert Tc == cost_type(F)
        @assert Tw == cost_type(W)
        return new{F, W, Tw, Tc}(f, w, f_max, w_max)
    end
end

ConstrainedCostOracle(f::F, w::W, w_max, f_max) where {F, W} = ConstrainedCostOracle{F, W, cost_type(F), cost_type(W)}(f, w, f_max, w_max)

oracle_model(ocl::ConstrainedCostOracle) = ConstrainedCost(oracle_model(ocl.f), oracle_model(ocl.w), ocl.w_max)

function oracle_stripe(cst::ConstrainedCost{F, W, Tw}, A::SparseMatrixCSC, args...; kwargs...) where {F, W, Tw}
    (m, n) = size(A)
    f = oracle_stripe(cst.f, A, args...; kwargs...)
    f_max = f(1, n + 1) + Inf #TODO this is a reasonable hack. The safest alternatitive is to introduce a wrapper numerical type which encodes infeasibility.
    w = oracle_stripe(cst.w, A, args...; kwargs...)
    w_max = cst.w_max
    return ConstrainedCostOracle(f, w, f_max, w_max)
end

@inline function (ocl::ConstrainedCostOracle)(j::Ti, j′::Ti, k...) where {Ti}
    if ocl.w(j, j′, k...) <= ocl.w_max
        return ocl.f(j, j′, k...)
    else
        return ocl.f_max
    end
end

bound_stripe(A::SparseMatrixCSC, K, cst::ConstrainedCost) = bound_stripe(A, K, ocl.f)
bound_stripe(A::SparseMatrixCSC, K, ocl::ConstrainedCostOracle) = bound_stripe(A, K, ocl.f)