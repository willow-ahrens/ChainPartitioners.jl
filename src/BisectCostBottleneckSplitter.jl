struct BisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::BisectCostBottleneckSplitter, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        f = oracle_stripe(SparseHint(), method.f, A, args...; kwargs...)
        ϵ = method.ϵ

        #=
            search returns the largest j′ such that f(j, j′) <= c, returns
            max(j, j′_lo) - 1 if no such j′ can be found
        =#
        @inline function search(j, j′_lo, j′_hi, k, c)
            j′_lo = max(j, j′_lo)
            while j′_lo <= j′_hi
                j′ = fld2(j′_lo + j′_hi)
                if f(j, j′, k) <= c
                    j′_lo = j′ + 1
                else
                    j′_hi = j′ - 1
                end
            end
            return j′_hi
        end

        spl_lo = ones(Int, K + 1)
        spl_lo[K + 1] = n + 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        spl = undefs(Int, K + 1)
        spl[1] = 1
        spl[K + 1] = n + 1

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            spl[1] = 1
            chk = true
            for k = 1 : K - 1
                spl[k + 1] = search(spl[k], spl_lo[k + 1], spl_hi[k + 1], k, c)
                if spl[k + 1] < spl[k]
                    chk = false
                    spl[k + 1 : K] .= spl[k]
                    break
                end
            end
            if chk && f(spl[K], spl[K + 1], K) <= c
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
                spl_lo .= spl
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

struct FlipBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::FlipBisectCostBottleneckSplitter, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        f = oracle_stripe(SparseHint(), method.f, A, args...; kwargs...)
        ϵ = method.ϵ

        #=
            search returns the smallest j′ such that f(j, j′) <= c, returns
            j′_hi + 1 if no such j′ can be found
        =#
        @inline function search(j, j′_lo, j′_hi, k, c)
            j′_lo = max(j, j′_lo)
            while j′_lo <= j′_hi
                j′ = fld2(j′_lo + j′_hi)
                if f(j, j′, k) <= c
                    j′_hi = j′ - 1
                else
                    j′_lo = j′ + 1
                end
            end
            return j′_lo
        end

        spl_lo = ones(Int, K + 1)
        spl_lo[K + 1] = n + 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        spl = undefs(Int, K + 1)
        spl[1] = 1
        spl[K + 1] = n + 1

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            spl[1] = 1
            chk = true
            for k = 1 : K - 1
                spl[k + 1] = search(spl[k], spl_lo[k + 1], spl_hi[k + 1], k, c)
                if spl[k + 1] > n + 1
                    chk = false
                    spl[k + 1 : K] .= n + 1
                    break
                end
            end
            if chk && f(spl[K], spl[K + 1], K) <= c
                c_hi = c
                spl_lo .= spl
            else
                c_lo = c
                spl_hi .= spl
            end
        end
        return SplitPartition(K, spl_lo)
    end
end