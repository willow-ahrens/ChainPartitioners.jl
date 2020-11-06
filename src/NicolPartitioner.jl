struct NicolPartitioner{F}
    f::F
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::NicolPartitioner, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        f = oracle_stripe(method.f, A, K, args...; kwargs...)

        #=
            search returns the largest j′ such that
            f(j, j′) <= c
        =#
        @inline function search(j, j′_lo, j′_hi, k, c)
            j′_lo = max(j, j′_lo)
            let j′
                while j′_lo <= j′_hi
                    j′ = fld2(j′_lo + j′_hi)
                    if f(j, j′, k) <= c
                        j′_lo = j′ + 1
                    else # f(j, j′) > c
                        j′_hi = j′ - 1
                    end
                end
                return j′_hi
            end
        end

        spl_lo = ones(Int, K + 1)

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        spl = undefs(Int, K + 1)
        spl[1] = 1

        c_lo, c_hi = bound_stripe(A, K, args..., f)

        for k = 1:K
            #invariants
            #c_hi is the least known achievable cost
            #j′_hi is the leftmost known split point to achieve c_hi
            #c_lo is the highest cost that might be achievable
            #j′_lo is the rightmost split point that might achieve a cost less than c_hi (at it's bottleneck)
            #wish to find the rightmost $j′$ which is unachievable at it's cost.
            j′_hi = spl_hi[k + 1]
            j′_lo = spl_lo[k + 1]
            while j′_lo <= j′_hi
                j′ = fld2(j′_lo + j′_hi)
                c = f(spl[k], j′, k)
                # So we picked j′. Million dollar question is: can we achieve spl[k + 1] at the cost of c?
                if c_lo <= c < c_hi #just try it
                    spl[k + 1] = j′
                    chk = true
                    for k′ = k + 1 : K
                        spl[k′ + 1] = search(spl[k′], spl_lo[k′ + 1], spl_hi[k′ + 1], k′, c)
                        if spl[k′ + 1] < max(spl[k′], spl_lo[k′ + 1])
                            chk = false
                            spl[k′ + 1 : end] .= spl[k′]
                            break
                        end
                    end
                    if chk && spl[K + 1] == n + 1 #yes, we can achieve spl[k + 1] at the cost of c.
                        c_hi = c #thus, we have a lower achievable cost for c_hi
                        j′_hi = j′ - 1
                        @assert spl_hi >= spl
                        spl_hi .= spl #record the split points which achieved c_hi
                    else #no we cannot achieve spl[k + 1] at the cost of c.
                        c_lo = c #since c is not achievable, no cost below c can be achievable
                        j′_lo = j′ + 1
                        @assert spl_lo <= spl
                        spl_lo .= spl #these split points cannot achieve an improvement on c_hi
                    end
                elseif c >= c_hi #c is a known achievable value, so unless we can get smaller this isn't worth investigating.
                    j′_hi = j′ - 1
                else #c is known to be unachievable, so unless we can get more this isn't worth investigating.
                    j′_lo = j′ + 1
                end
            end

            if j′_hi < max(spl[k], spl_lo[k + 1])
                break
            end

            spl[k + 1] = j′_hi
        end

        return SplitPartition(K, spl_hi)
    end
end

struct FlipNicolPartitioner{F}
    f::F
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::FlipNicolPartitioner, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        f = oracle_stripe(method.f, A, K, args...; kwargs...)

        #=
            search returns the largest j′ such that
            f(j, j′) <= c
        =#
        @inline function search(j, j′_lo, j′_hi, k, c)
            j′_lo = max(j, j′_lo)
            @assert j′_lo <= j′_hi
            let j′
                while j′_lo <= j′_hi
                    j′ = fld2(j′_lo + j′_hi)
                    if f(j, j′, k) <= c
                        j′_hi = j′ - 1
                    else # f(j, j′) > c
                        j′_lo = j′ + 1
                    end
                end
                return j′_lo
            end
        end

        spl_lo = ones(Int, K + 1)

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        spl = undefs(Int, K + 1)
        spl[1] = 1

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        for k = 1:K
            #invariants
            #c_hi is the least known achievable cost
            #j′_hi is the leftmost known split point to achieve c_hi
            #c_lo is the highest cost that might be achievable
            #j′_lo is the rightmost split point that might achieve a cost less than c_hi (at it's bottleneck)
            #wish to find the rightmost $j′$ which is unachievable at it's cost.
            j′_hi = spl_hi[k + 1]
            j′_lo = max(spl[k], spl_lo[k + 1])
            while j′_lo <= j′_hi
                j′ = fld2(j′_lo + j′_hi)
                c = f(spl[k], j′, k)
                # So we picked j′. Million dollar question is: can we achieve spl[k + 1] at the cost of c?
                if c_lo <= c < c_hi #just try it
                    spl[k + 1] = j′
                    chk = true
                    for k′ = k + 1 : K
                        spl[k′ + 1] = search(spl[k′], spl_lo[k′ + 1], spl_hi[k′ + 1], k′, c)
                        if spl[k′ + 1] > spl_hi[k′ + 1]
                            chk = false
                            spl[k′ + 1 : end] .= spl_hi[k′ + 1 : end]
                            break
                        end
                    end
                    if chk #yes, we can achieve spl[k + 1] at the cost of c.
                        c_hi = c #thus, we have a lower achievable cost for c_hi
                        j′_lo = j′ + 1
                        @assert spl_lo <= spl
                        spl_lo .= spl
                    else #no we cannot achieve spl[k + 1] at the cost of c.
                        c_lo = c #since c is not achievable, no cost below c can be achievable
                        j′_hi = j′ - 1
                        @assert spl_hi >= spl
                        spl_hi .= spl
                    end
                elseif c >= c_hi #c is a known achievable value, so unless we can get smaller this isn't worth investigating.
                    j′_lo = j′ + 1
                else #c is known to be unachievable, so unless we can get more this isn't worth investigating.
                    j′_hi = j′ - 1
                end
            end

            if j′_lo > spl_hi[k + 1]
                break
            end

            spl[k + 1] = j′_lo 
        end

        spl_lo[K + 1] = n + 1 
        return SplitPartition(K, spl_lo)
    end
end