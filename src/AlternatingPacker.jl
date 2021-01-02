struct DisjointPacker{Mtd, Mtd′}
    mtd::Mtd
    mtd′::Mtd′
end

function pack_plaid(A::SparseMatrixCSC, method::DisjointPacker; kwargs...)
    Φ = pack_stripe(A, method.mtd; kwargs...)
    Π = pack_stripe(PermutedDimsArray(A, (2, 1)), method.mtd′, Φ; kwargs...)
    return (Π, Φ)
end

struct AlternatingPacker{Mtds}
    mtds::Mtds
end

AlternatingPacker(mtds...) = AlternatingPacker{typeof(mtds)}(mtds)

function pack_plaid(A::SparseMatrixCSC, method::AlternatingPacker; adj_A = nothing, kwargs...)
    if adj_A === nothing
        adj_A = adjointpattern(A)
    end
    Φ = pack_stripe(A, method.mtds[1]; adj_A=adj_A, kwargs...)
    Π = pack_stripe(adj_A, method.mtds[2], Φ; adj_A=A, kwargs...)
    for (i, mtd) in enumerate(method.mtds[3:end])
        if isodd(i)
            Φ = pack_stripe(A, mtd, Π; adj_A=adj_A, kwargs...)
        else
            Π = pack_stripe(adj_A, mtd, Φ; adj_A=A, kwargs...)
        end
    end
    return (Π, Φ)
end

struct SymmetricPacker{Mtds}
    mtds::Mtds
end

SymmetricPacker(mtds...) = new{typeof(mtds)}(mtds)

function pack_plaid(A::SparseMatrixCSC, method::SymmetricPacker; adj_A = nothing, kwargs...)
    if adj_A === nothing
        adj_A = adjointpattern(A)
    end
    Π = pack_stripe(A, method.mtds[1]; adj_A=adj_A, kwargs...)
    for (i, mtd) in enumerate(method.mtds[2:end])
        if isodd(i)
            Π = pack_stripe(A, mtd, Π; adj_A=adj_A, kwargs...)
        else
            Π = pack_stripe(adj_A, mtd, Π; adj_A=A, kwargs...)
        end
    end
    return (Π, Π)
end