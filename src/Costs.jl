abstract type AbstractCostOracle end

function oracle_stripe(mdl, A, Π; kwargs...)
    return oracle_stripe(mdl, A; kwargs...)
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

function bottleneck_value(A, Π, Φ, mdl)
    return bottleneck_value(A, Π, Φ, oracle_stripe(mdl, A, Π))
end

function bottleneck_value(A, Π, Φ, mdl::AbstractCostOracle)
    cst = 0
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = max(cst, mdl(j, j′, k))
    end
    return cst
end

function bottleneck_value(A, Φ, mdl)
    return bottleneck_value(A, Φ, oracle_stripe(mdl, A))
end

function bottleneck_value(A, Φ::SplitPartition, mdl::AbstractCostOracle)
    cst = 0
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = max(cst, mdl(j, j′, k))
    end
    return cst
end

function total_value(A, Π, Φ, mdl)
    return total_value(A, Π, Φ, oracle_stripe(mdl, A, Π))
end

function total_value(A, Π, Φ, mdl::AbstractCostOracle)
    cst = 0
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = cst + mdl(j, j′, k)
    end
    return cst
end

function total_value(A, Φ, mdl)
    return total_value(A, Φ, oracle_stripe(mdl, A))
end

function total_value(A, Φ::SplitPartition, mdl::AbstractCostOracle)
    cst = 0
    for k = 1:Φ.K
        j = Φ.spl[k]
        j′ = Φ.spl[k + 1]
        cst = cst + mdl(j, j′, k)
    end
    return cst
end