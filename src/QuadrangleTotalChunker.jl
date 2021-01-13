struct ConvexTotalChunker{F}
    f::F
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConvexTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        stk = Stack{Tuple{Ti, Ti}}(n)
        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[n + 1] = zero(cost_type(f))
        cst′(j, j′) = cst[j′] + f(j, j′)

        push!(stk, (n + 1, 1))
        for j = n:-1:1
            (j′, h) = first(stk)
            if cst′(j, j + 1) ≥ cst′(j, j′)
                cst[j] = cst′(j, j′)
                spl[j] = j′
                if h == j
                    pop!(stk)
                end
            else
                cst[j] = cst′(j, j + 1)
                spl[j] = j + 1
                while !isempty(stk) && ((j′, h) = first(stk); (cst′(h, j + 1) < cst′(h, j′)))
                    pop!(stk)
                end
                if isempty(stk)
                    push!(stk, (j + 1, 1))
                else
                    (j′, h) = first(stk)
                    #@assert cst′(h, j + 1) >= cst′(h, j′)
                    #h′ = j - 1
                    #while h′ > 0 && cst′(h′, j + 1) < cst′(h′, j′) h′ -= 1 end
                    #h′ += 1
                    h′_lo = h + 1
                    h′_hi = j
                    while h′_lo <= h′_hi
                        h′ = fld2(h′_lo + h′_hi)
                        if cst′(h′, j + 1) < cst′(h′, j′)
                            h′_hi = h′ - 1
                        else
                            h′_lo = h′ + 1
                        end
                    end
                    #@assert h′ == h′_lo
                    h′ = h′_lo

                    if h′ < j
                        push!(stk, (j + 1, h′))
                    end
                end
            end
        end

        K = 0
        j = 1
        while j != n + 1
            j′ = spl[j]
            K += 1
            spl[K] = j
            j = j′
        end
        spl[K + 1] = j
        resize!(spl, K + 1)
        return SplitPartition{Ti}(K, spl) 
    end
end