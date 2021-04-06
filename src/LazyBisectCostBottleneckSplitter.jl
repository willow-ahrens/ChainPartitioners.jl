struct LazyBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

#TODO this file needs to get redistribued, and we should use step oracles for comm costs.

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{F}, args...; kwargs...) where {Tv, Ti, F}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A, args...)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                f(1, 1, 1)

                for j′ = 2:n + 1
                    if Step(f, Same(), Next(), Same())(j, j′, k) > c
                        while true
                            if k == K
                                return false
                            end
                            spl[k + 1] = j′ - 1
                            j = j′ - 1
                            k += 1
                            if f(j, j′, k) <= c
                                break
                            end
                        end
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., method.f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(1, 1, k))
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

struct LazyFlipBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

#TODO this file needs to get redistribued, and we should use step oracles for comm costs.

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyFlipBisectCostBottleneckSplitter{F}, args...; kwargs...) where {Tv, Ti, F}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A, args...)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                f(1, 1, 1)

                for j′ = 2:n + 1
                    if Step(f, Same(), Next(), Same())(j, j′, k) <= c
                        while true
                            if k == K
                                spl[K + 1] = n + 1
                                return true
                            end
                            spl[k + 1] = j′
                            j = j′
                            k += 1
                            if f(j, j′, k) > c
                                break
                            end
                        end
                    end
                end
                return false
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., method.f) ./ 1

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end