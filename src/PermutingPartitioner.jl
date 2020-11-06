struct PermutingPartitioner{Prm, Prt}
    prm::Prm
    prt::Prt
end

function partition_plaid(A::SparseMatrixCSC, K, method::PermutingPartitioner; adj_A = nothing, kwargs...)
    @inbounds begin
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        σ, τ = permute_plaid(A, method.prm; adj_A = adj_A)
        B = A[perm(σ), perm(τ)]
        Π, Φ = partition_plaid(B, K, method.prt; adj_A = adj_A)
        return (σ[Π], τ[Φ])
    end
end

function partition_stripe(A::SparseMatrixCSC, K, method::PermutingPartitioner; kwargs...)
    @inbounds begin
        τ = permute_stripe(A, method.prm)
        B = A[:, perm(τ)]
        Φ = partition_stripe(B, K, method.prt)
        return τ[Φ]
    end
end

function partition_stripe(A::SparseMatrixCSC, K, Π, method::PermutingPartitioner; kwargs...)
    @inbounds begin
        τ = permute_stripe(A, method.prm)
        B = A[:, perm(τ)]
        Φ = partition_stripe(B, K, Π, method.prt)
        return τ[Φ]
    end
end