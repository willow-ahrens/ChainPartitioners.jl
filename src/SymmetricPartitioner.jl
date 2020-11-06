struct SymmetricPartitioner{Mtd}
    mtd::Mtd
end

function partition_plaid(A::SparseMatrixCSC, k, method::SymmetricPartitioner; kwargs...)
    Π = partition_stripe(A, k, method.mtd; kwargs...)
    return (Π, Π)
end