abstract type AbstractDynamicSplitter{F} end

struct DynamicBottleneckSplitter{F} <: AbstractDynamicSplitter{F}
    f::F
end

@inline _dynamic_splitter_combine(::DynamicBottleneckSplitter) = max

struct DynamicTotalSplitter{F} <: AbstractDynamicSplitter{F}
    f::F
end

@inline _dynamic_splitter_combine(::DynamicTotalSplitter) = +

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::AbstractDynamicSplitter, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)
        g = _dynamic_splitter_combine(method)

        ptr = zeros(Ti, K, n + 1)
        cst = fill(typemax(cost_type(f)), K, n + 1)

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
        return unravel_splits(K, n, ptr)
    end
end

function unravel_splits(K, n, ptr)
    spl = zeros(eltype(ptr), K + 1)
    spl[end] = n + 1
    for k = K:-1:1
        spl[k] = ptr[k, spl[k + 1]]
    end

    return SplitPartition(K, spl)
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::AbstractDynamicSplitter{<:ConstrainedCost}, args...) where {Tv, Ti}
    begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)
        w = f.w
        w_max = method.f.w_max
        g = _dynamic_splitter_combine(method)

        j_lo = undefs(Ti, K + 1)
        j′ = n + 1
        j_lo[K + 1] = n + 1
        for k = K:-1:1
            j = j′
            while j - 1 >= 1 && w(j - 1, j′, k) <= w_max
                j -= 1
            end
            j_lo[k] = j
            j′ = j
        end

        j′_hi = undefs(Ti, K)
        j = 1
        for k = 1:K
            j′ = j
            while j′ + 1 <= n + 1 && w(j, j′ + 1, k) <= w_max
                j′ += 1
            end
            j′_hi[k] = j′
            j = j′
        end

        if j′_hi[K] < n + 1
            spl = ones(Ti, K + 1)
            spl[end] = n + 1
            #TODO throw(ArgumentError("infeasible"))
            return SplitPartition(K, spl)
        end

        pos = undefs(Ti, K + 1)
        pos[1] = 1
        for k = 1:K
            pos[k + 1] = pos[k] + (1 + j′_hi[k] - j_lo[k + 1])
        end

        ptr = zeros(Ti, pos[end])
        cst = undefs(cost_type(f), pos[end])

        for j′ = j_lo[2] : j′_hi[1]
            i′ = j′ - j_lo[2]
            cst[pos[1] + i′] = g(zero(cost_type(f)), f(1, j′, 1))
            ptr[pos[1] + i′] = 1
        end

        for k = 2:K
            j₀ = j_lo[k]
            for j′ = j_lo[k + 1] : j′_hi[k]
                while w(j₀, j′, k) > w_max #TODO can this run over the end?
                    j₀ += 1
                end
                i′ = j′ - j_lo[k + 1]
                i = j₀ - j_lo[k]
                cst[pos[k] + i′] = g(cst[pos[k - 1] + i], f(j₀, j′, k))
                ptr[pos[k] + i′] = j₀
                for j = j₀ + 1 : min(j′, j′_hi[k - 1])
                    i = j - j_lo[k]
                    c′ = g(cst[pos[k - 1] + i], f(j, j′, k))
                    if c′ <= cst[pos[k] + i′]
                        cst[pos[k] + i′] = c′ 
                        ptr[pos[k] + i′] = j
                    end
                end
            end
        end
        return unravel_constrained_splits(K, n, pos, ptr, j_lo)
    end
end

function unravel_constrained_splits(K, n, pos, ptr, j_lo)
    spl = zeros(eltype(ptr), K + 1)
    spl[end] = n + 1
    for k = K:-1:1
        spl[k] = ptr[pos[k] + spl[k + 1] - j_lo[k + 1]]
    end

    return SplitPartition(K, spl)
end