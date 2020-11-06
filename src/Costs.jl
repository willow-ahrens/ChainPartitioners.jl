abstract type AbstractCostOracle end

function oracle_stripe(mdl, A, K, Π; kwargs...)
    return oracle_stripe(mdl, A, K; kwargs...)
end

function bound_stripe(A, K, Π, mdl::AbstractCostOracle)
    return bound_stripe(A, K, Π, mdl.mdl)
end

function bound_stripe(A, K, mdl::AbstractCostOracle)
    return bound_stripe(A, K, mdl.mdl)
end

function bound_stripe(A, K, Π, mdl)
    return bound_stripe(A, K, mdl)
end

function bottleneck_plaid(A, K, Π, Φ, mdl)
    return bottleneck_stripe(A, K, Φ, mdl)
end

function bottleneck_stripe(A, K, Φ::SplitPartition, mdl::AbstractCostOracle)
    cst = 0
    for k = 1:K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = max(cst, mdl(j, j′, k))
    end
    return cst
end