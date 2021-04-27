struct DisjointPartitioner{Mtd, Mtd′}
    mtd::Mtd
    mtd′::Mtd′
end

function partition_plaid(A::SparseMatrixCSC, K, method::DisjointPartitioner; kwargs...)
    Φ = partition_stripe(A, K, method.mtd; kwargs...)
    Π = partition_stripe(PermutedDimsArray(A, (2, 1)), K, method.mtd′, Φ; kwargs...)
    return (Π, Φ)
end

struct AlternatingPartitioner{Mtds}
    mtds::Mtds
end

AlternatingPartitioner(mtds...) = AlternatingPartitioner{typeof(mtds)}(mtds)

function partition_plaid(A::SparseMatrixCSC, K, method::AlternatingPartitioner; adj_A = nothing, kwargs...)
    if adj_A === nothing
        adj_A = adjointpattern(A)
    end
    Φ = partition_stripe(A, K, method.mtds[1]; adj_A=adj_A, kwargs...)
    Π = partition_stripe(adj_A, K, method.mtds[2], Φ; adj_A=A, kwargs...)
    for (i, mtd) in enumerate(method.mtds[3:end])
        if isodd(i)
            Φ = partition_stripe(A, K, mtd, Π; adj_A=adj_A, kwargs...)
        else
            Π = partition_stripe(adj_A, K, mtd, Φ; adj_A=A, kwargs...)
        end
    end
    return (Π, Φ)
end

struct AlternatingNetPartitioner{Hint, Mtds}
    hint::Hint
    mtds::Mtds
end

AlternatingNetPartitioner(mtds...) = AlternatingNetPartitioner{NoHint, typeof(mtds)}(NoHint(), mtds)
AlternatingNetPartitioner(hint::Hint, mtds...) where {Hint<:AbstractHint} = AlternatingNetPartitioner{Hint, typeof(mtds)}(hint, mtds)

function partition_plaid(A::SparseMatrixCSC, K, method::AlternatingNetPartitioner; adj_A = nothing, net = nothing, kwargs...)
    if adj_A === nothing
        adj_A = adjointpattern(A)
    end
    if net === nothing
        net = netcount(method.hint, A; kwargs...)
    end
    Φ = partition_stripe(A, K, method.mtds[1]; net=net, adj_A=adj_A, kwargs...)
    Π = partition_stripe(adj_A, K, method.mtds[2], Φ; adj_A=A, kwargs...)
    for (i, mtd) in enumerate(method.mtds[3:end])
        if isodd(i)
            Φ = partition_stripe(A, K, mtd, Π; net=net, adj_A=adj_A, kwargs...)
        else
            Π = partition_stripe(adj_A, K, mtd, Φ; adj_A=A, adj_net = net, kwargs...)
        end
    end
    return (Π, Φ)
end

struct SymmetricPartitioner{Mtds}
    mtds::Mtds
    function SymmetricPartitioner{Mtds}(mtds::Mtds) where {Mtds}
        return new{Mtds}(mtds)
    end
end

SymmetricPartitioner(mtd) = SymmetricPartitioner{Tuple{typeof(mtd)}}((mtd,))
SymmetricPartitioner(mtds...) = SymmetricPartitioner{typeof(mtds)}(mtds)

function partition_plaid(A::SparseMatrixCSC, K, method::SymmetricPartitioner; adj_A = nothing, kwargs...)
    if adj_A === nothing
        adj_A = adjointpattern(A)
    end
    Π = partition_stripe(A, K, method.mtds[1]; adj_A=adj_A, kwargs...)
    for (i, mtd) in enumerate(method.mtds[2:end])
        if isodd(i)
            Π = partition_stripe(A, K, mtd, Π; adj_A=adj_A, kwargs...)
        else
            Π = partition_stripe(adj_A, K, mtd, Π; adj_A=A, kwargs...)
        end
    end
    return (Π, Π)
end