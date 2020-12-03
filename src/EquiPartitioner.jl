struct EquiSplitter end

function partition_stripe(A::AbstractMatrix, K, ::EquiSplitter, args...; kwargs...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        return SplitPartition(K, map(k-> (k - 1) * fld(n, K) + min(n % K, k - 1) + 1, 1:(K + 1)))
    end
end

struct EquiChunker
    w::Int
end

function pack_stripe(A::AbstractMatrix, method::EquiChunker, args...; kwargs...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        w = method.w

        return SplitPartition(cld(n, K), collect(1:w:n))
    end
end