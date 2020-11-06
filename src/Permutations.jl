abstract type AbstractPermutation{Ti} end

struct DomainPermutation{Ti} <: AbstractPermutation{Ti}
    prm::Vector{Ti}
end

struct MapPermutation{Ti} <: AbstractPermutation{Ti}
    asg::Vector{Ti}
end

function Base.:(==)(σ::MapPermutation, σ′::MapPermutation)
    return σ.asg == σ′.asg
end

function Base.:(==)(σ::DomainPermutation, σ′::DomainPermutation)
    return σ.prm == σ′.prm
end

function Base.convert(T::Type{<:DomainPermutation}, σ::MapPermutation{Ti}) where {Ti}
    return T(invperm(σ.asg))
end

function Base.convert(T::Type{<:MapPermutation}, σ::DomainPermutation{Ti}) where {Ti}
    return T(invperm(σ.prm))
end

function perm(σ::AbstractPermutation)
    return convert(DomainPermutation, σ).prm
end

struct IdentityPermuter end

function permute_stripe(A::AbstractMatrix, ::IdentityPermuter; kwargs...)
    (m, n) = size(A)
    return DomainPermutation(collect(1:n))
end

function permute_plaid(A::AbstractMatrix, ::IdentityPermuter; kwargs...)
    (m, n) = size(A)
    return (DomainPermutation(collect(1:m)), DomainPermutation(collect(1:n)))
end

function Base.getindex(σ::AbstractPermutation{Ti}, Π::SplitPartition) where {Ti}
    @inbounds return DomainPartition(Π.K, (1:(Π.spl[end] - 1))[perm(σ)], Π.spl)
end

function Base.getindex(σ::AbstractPermutation, Π::DomainPartition)
    @inbounds return DomainPartition(Π.K, Π.prm[perm(σ)], Π.spl)
end

function Base.getindex(σ::AbstractPermutation, Π::MapPartition)
    @inbounds return MapPartition(Π.K, Π.asg[convert(MapPermutation, σ).asg])
end