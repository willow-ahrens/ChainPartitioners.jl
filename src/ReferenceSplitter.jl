struct ReferenceTotalSplitter{F}
    f::F
end

partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ReferenceTotalSplitter{F}, args...; kwargs...) where {Tv, Ti, F} =
    invoke(partition_stripe, Tuple{SparseMatrixCSC{Tv, Ti}, Any, DynamicTotalSplitter, Vararg}, A, K, DynamicTotalSplitter(method.f), args...; kwargs...)

struct ReferenceBottleneckSplitter{F}
    f::F
end

partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ReferenceBottleneckSplitter{F}, args...; kwargs...) where {Tv, Ti, F} =
    invoke(partition_stripe, Tuple{SparseMatrixCSC{Tv, Ti}, Any, DynamicBottleneckSplitter, Vararg}, A, K, DynamicBottleneckSplitter(method.f), args...; kwargs...)


struct ReferenceTotalChunker{F}
    f::F
end

pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ReferenceTotalChunker{F}, args...; kwargs...) where {Tv, Ti, F} =
    invoke(pack_stripe, Tuple{SparseMatrixCSC{Tv, Ti}, DynamicTotalChunker, Vararg}, A, DynamicTotalChunker(method.f), args...; kwargs...)