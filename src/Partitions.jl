abstract type AbstractPartition{Ti} end

struct SplitPartition{Ti} <: AbstractPartition{Ti}
    K::Int
    spl::Vector{Ti}
end

struct DomainPartition{Ti} <: AbstractPartition{Ti}
    K::Int
    prm::Vector{Ti} 
    spl::Vector{Ti}
end

struct MapPartition{Ti} <: AbstractPartition{Ti}
    K::Int
    asg::Vector{Ti}
end

function Base.:(==)(Π::SplitPartition, Π′::SplitPartition)
    return Π.spl == Π′.spl
end

function Base.:(==)(Π::DomainPartition, Π′::DomainPartition)
    return Π.prm == Π′.prm && Π.spl == Π′.spl
end

function Base.:(==)(Π::MapPartition, Π′::MapPartition)
    return Π.K == Π′.K && Π.asg == Π′.asg
end

function Base.convert(T::Type{<:DomainPartition}, Π::SplitPartition{Ti}) where {Ti}
    prm = collect(1:(Π.spl[end]-1))
    return T(Π.K, prm, Π.spl)
end

function Base.convert(T::Type{<:DomainPartition}, Π::MapPartition{Ti}) where {Ti}
    n = length(Π.asg)
    spl = zeros(Ti, Π.K + 1)
    prm = undefs(Ti, n)
    for j = 1:n
        k = Π.asg[j]
        spl[k + 1] += 1
    end
    q = 1
    for k = 1:(Π.K + 1)
        (spl[k], q) = (q, q + spl[k])
    end
    for j = 1:n
        k = Π.asg[j]
        prm[spl[k + 1]] = j
        spl[k + 1] += 1
    end
    return T(Π.K, prm, spl)
end

function Base.convert(T::Type{<:MapPartition}, Π::SplitPartition{Ti}) where {Ti}
    asg = undefs(Ti, Π.spl[end]-1)
    for k = 1:Π.K
        for j = Π.spl[k]:(Π.spl[k + 1] - 1)
            asg[j] = k
        end
    end
    return T(Π.K, asg)
end

function Base.convert(T::Type{<:MapPartition}, Π::DomainPartition{Ti}) where {Ti}
    asg = undefs(Ti, Π.spl[end]-1)
    for k = 1:Π.K
        for q = Π.spl[k]:(Π.spl[k + 1] - 1)
            j = Π.prm[q]
            asg[j] = k
        end
    end
    return T(Π.K, asg)
end