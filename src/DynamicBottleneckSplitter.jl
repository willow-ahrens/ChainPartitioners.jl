abstract type AbstractDynamicSplitter{F} end

struct DynamicBottleneckSplitter{F} <: AbstractDynamicSplitter{F}
    f::F
end

@inline _dynamic_splitter_aggregate(::DynamicBottleneckSplitter) = max

struct DynamicTotalSplitter{F} <: AbstractDynamicSplitter{F}
    f::F
end

@inline _dynamic_splitter_aggregate(::DynamicTotalSplitter) = sum

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::DynamicBottleneckSplitter, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)
        g = _dynamic_splitter_aggregate(method)

        ptr = zeros(Ti, K + 1, n + 1)
        cst = fill(Inf, K, n + 1)

        for j′ = 1:n + 1
            cst[1, j′] = f(1, j′, 1)
            ptr[1, j′] = 1
            for j = 1:j′
                for k = 2:K
                    c_lo = f(j, j′, k)
                    if g(cst[k - 1, j], c_lo) <= cst[k, j′]
                        cst[k, j′] = g(cst[k - 1, j], c_lo)
                        ptr[k, j′] = j
                    end
                end
            end
        end

        spl = zeros(Ti, K + 1)
        spl[end] = n + 1
        for k = K:-1:1
            spl[k] = ptr[k, spl[k + 1]]
        end

        return SplitPartition(K, spl)
    end
end