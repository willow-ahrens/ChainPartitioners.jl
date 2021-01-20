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



struct Extended{T}
    i::Bool
    x::T
end

infinity(::Type{Float16}) = Inf16
infinity(::Type{Float32}) = Inf32
infinity(::Type{Float64}) = Inf64
infinity(::Type{Extended{T}}) where {T} = Extended(true, zero(T))
infinity(x) = infinity(typeof(x))
infinity(T::Type) = Extended(true, zero(T))

extend(T::Type) = typeof(infinity(T))
#TODO use convert here
extend(x) = (x isa extend(typeof(x))) ? x : Extended(false, x)

#Base.zero(::Extended{T}) where {T} = extend(zero(T))
Base.zero(::Type{Extended{T}}) where {T} = extend(zero(T))
Base.:+(a::Extended, b::Extended) = Extended(a.i | b.i, a.x + b.x)
Base.:*(a::Extended, b::Extended) = Extended(a.i | b.i, a.x * b.x)
Base.typemax(::Type{Extended{T}}) where {T} = infinity(T)
Base.typemin(::Type{Extended{T}}) where {T} = extend(typemin(T))
Base.:<(a::Extended, b::Extended) = (!a.i && b.i) || ((!a.i && !b.i) && (a.x < b.x))
Base.:(==)(a::Extended, b::Extended) = (a.i && b.i) || ((!a.i && !b.i) && (a.x == b.x))
Base.isapprox(a::Extended, b::Extended; kwargs...) = (a.i && b.i) || ((!a.i && !b.i) && isapprox(a.x, b.x; kwargs...))

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

cost_type(::Type{ConstrainedCost{F, W, Tw}}) where {F, W, Tw} = extend(cost_type(F))

struct ConstrainedCostOracle{F, W, Tw} <: AbstractOracleCost{ConstrainedCost{F, W, Tw}}
    f::F
    w::W
    w_max::Tw
    function ConstrainedCostOracle{F, W, Tw}(f::F, w::W, w_max) where {F, W, Tw}
        @assert Tw == cost_type(W)
        return new{F, W, Tw}(f, w, w_max)
    end
end

ConstrainedCostOracle(f::F, w::W, w_max) where {F, W} = ConstrainedCostOracle{F, W, cost_type(W)}(f, w, w_max)

oracle_model(ocl::ConstrainedCostOracle) = ConstrainedCost(oracle_model(ocl.f), oracle_model(ocl.w), ocl.w_max)

function oracle_stripe(cst::ConstrainedCost{F, W, Tw}, A::SparseMatrixCSC, args...; kwargs...) where {F, W, Tw}
    (m, n) = size(A)
    f = oracle_stripe(cst.f, A, args...; kwargs...)
    w = oracle_stripe(cst.w, A, args...; kwargs...)
    w_max = cst.w_max
    return ConstrainedCostOracle(f, w, w_max)
end

@inline function (ocl::ConstrainedCostOracle)(j::Ti, j′::Ti, k...) where {Ti}
    if ocl.w(j, j′, k...) <= ocl.w_max
        return extend(ocl.f(j, j′, k...))
    else
        return infinity(cost_type(ocl.f))
    end
end

#TODO not sure about these two...
bound_stripe(A::SparseMatrixCSC, K, cst::ConstrainedCost) = bound_stripe(A, K, ocl.f)
bound_stripe(A::SparseMatrixCSC, K, ocl::ConstrainedCostOracle) = bound_stripe(A, K, ocl.f)

struct Infeasible end

struct Feasible end

Base.:<(::Feasible, ::Feasible) = false
Base.:<(::Feasible, ::Infeasible) = true
Base.:<(::Infeasible, ::Infeasible) = false
Base.:<(::Infeasible, ::Feasible) = false
Base.:(==)(::Feasible, ::Feasible) = true
Base.:(==)(::Feasible, ::Infeasible) = false
Base.:(==)(::Infeasible, ::Feasible) = false
Base.:(==)(::Infeasible, ::Infeasible) = true

struct FeasibleCost end

@inline (::FeasibleCost)(j, j′, k...) = Feasible()
@inline cost_type(::Type{FeasibleCost}) = Feasible
oracle_model(::FeasibleCost) = FeasibleCost()
oracle_stripe(::FeasibleCost, ::SparseMatrixCSC; kwargs...) = FeasibleCost()
#bound_stripe(A::SparseMatrixCSC, K, ::FeasibleCost) = (Feasible(), Feasible()) #TODO ?

struct WidthCost{Ti} end

@inline (::WidthCost{Ti})(j, j′, k...) where {Ti} = Ti(j′ - j)
@inline cost_type(::Type{WidthCost{Ti}}) where {Ti} = Ti
oracle_model(::WidthCost{Ti}) where {Ti} = WidthCost{Ti}()
oracle_stripe(::WidthCost{Ti}, ::SparseMatrixCSC; kwargs...) where {Ti} = WidthCost{Ti}()
#bound_stripe(A::SparseMatrixCSC, K, ::WidthCost) = (size(A)[2]/K, size(A)[2]) $TODO ?