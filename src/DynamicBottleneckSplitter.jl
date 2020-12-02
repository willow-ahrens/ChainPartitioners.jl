struct DynamicBottleneckSplitter{F}
    f::F
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::DynamicBottleneckSplitter, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        ptr = zeros(Ti, K + 1, n + 1)
        cst = fill(Inf, K, n + 1)

        for j′ = 1:n + 1
            cst[1, j′] = f(1, j′, 1)
            ptr[1, j′] = 1
            for j = 1:j′
                for k = 2:K
                    c_lo = f(j, j′, k)
                    if max(cst[k - 1, j], c_lo) <= cst[k, j′]
                        cst[k, j′] = max(cst[k - 1, j], c_lo)
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

#=
struct LeftistPartitioner{F}
    f::F
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LeftistPartitioner, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        Φ = partition_stripe(A, K, DynamicBottleneckSplitter(method.f), args...)
        f = oracle_stripe(method.f, A, args...)
        c = -Inf
        for k = 1:K
            c = max(c, f(Φ.spl[k], Φ.spl[k + 1], k))
        end

        spl = zeros(Ti, K + 1)
        spl[1] = 1
        for k = 1:K
            j = spl[k]
            j′ = j
            while j′ <= n + 1 && f(j, j′, k) <= c
                j′ += 1
            end
            spl[k + 1] = j′ - 1
        end

        return SplitPartition(K, spl)
    end
end

struct FlipLeftistPartitioner{F}
    f::F
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::FlipLeftistPartitioner, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)
        Φ = partition_stripe(A, K, DynamicBottleneckSplitter(method.f), args...)
        f = oracle_stripe(method.f, A, args...)
        c = -Inf
        for k = 1:K
            c = max(c, f(Φ.spl[k], Φ.spl[k + 1], k))
        end

        spl = zeros(Ti, K + 1)
        spl[1] = 1
        for k = 1:K
            j = spl[k]
            j′ = j
            while j′ < n + 1 && f(j, j′, k) > c
                j′ += 1
            end
            spl[k + 1] = j′
        end
        spl[K + 1] = n + 1

        return SplitPartition(K, spl)
    end
end
=#