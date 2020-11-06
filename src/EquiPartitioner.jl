struct EquiPartitioner end

function partition_stripe(A::AbstractMatrix, K, ::EquiPartitioner, args...; kwargs...) where {Tv, Ti}
    @inbounds begin

        (m, n) = size(A)
        return SplitPartition(K, map(k-> (k - 1) * fld(n, K) + min(n % K, k - 1) + 1, 1:(K + 1)))
    end
end