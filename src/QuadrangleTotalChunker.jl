struct ConvexTotalChunker{F}
    f::F
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConvexTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        ftr = Stack{Tuple{Ti, Ti}}(n)
        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[n + 1] = zero(cost_type(f))
        f′(j, j′) = cst[j′] + f(j, j′)
        chunk_convex!(cst, spl, f′, 1, n + 1, ftr)

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

function chunk_convex!(cst, ptr, f′, j₀, j₁, ftr)
    empty!(ftr)
    push!(ftr, (j₁, j₀))
    for j = j₁ - 1:-1:j₀
        (j′, h) = first(ftr)
        if f′(j, j + 1) ≥ f′(j, j′)
            cst[j] = f′(j, j′)
            ptr[j] = j′
            if h == j
                pop!(ftr)
            end
        else
            cst[j] = f′(j, j + 1)
            ptr[j] = j + 1
            while !isempty(ftr) && ((j′, h) = first(ftr); (f′(h, j + 1) < f′(h, j′)))
                pop!(ftr)
            end
            if isempty(ftr)
                push!(ftr, (j + 1, j₀))
            else
                (j′, h) = first(ftr)
                h′_lo = h + 1
                h′_hi = j
                while h′_lo <= h′_hi
                    h′ = fld2(h′_lo + h′_hi)
                    if f′(h′, j + 1) < f′(h′, j′)
                        h′_hi = h′ - 1
                    else
                        h′_lo = h′ + 1
                    end
                end
                h′ = h′_lo

                if h′ < j
                    push!(ftr, (j + 1, h′))
                end
            end
        end
    end
end

#=
function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConvexTotalChunker{<:ConstrainedCost}, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f.f, A, args...)
        w = oracle_stripe(method.f.w, A, args...)
        w_max = method.w_max

        stk = Stack{Tuple{Ti, Ti}}(n)
        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[n + 1] = zero(cost_type(f))
        cst′(j, j′) = cst[j′] + f(j, j′)
        σ = undefs(Ti, n)

        while j₀ > 0
            push!(stk, (j₁, j₀))
            for j = j₁ - 1:-1:j₀
                (j′, h) = first(stk)
                if cst′(j, j + 1) ≥ cst′(j, j′)
                    if cst′(j, j′) < cst[j]
                        cst[j] = cst′(j, j′)
                        spl[j] = j′
                    end
                    if h == j
                        pop!(stk)
                    end
                else
                    if cst′(j, j + 1) < cst[j]
                        cst[j] = cst′(j, j + 1)
                        spl[j] = j + 1
                    end
                    while !isempty(stk) && ((j′, h) = first(stk); (cst′(h, j + 1) < cst′(h, j′)))
                        pop!(stk)
                    end
                    if isempty(stk)
                        push!(stk, (j + 1, j₀))
                    else
                        (j′, h) = first(stk)
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
                        h′ = h′_lo

                        if h′ < j
                            push!(stk, (j + 1, h′))
                        end
                    end
                end
            end
            #j₀′ = j₀
            #j₁′ = j₁
            for j′ = j₁:-1:j₀
                while j > 1 && w(j - 1, j′) < w_max
                    j -= 1
                end
                σ[j′] = j
            end
            cst′′(j, j′) = cst′(σ[j], j′)
            for j′ = j₀:j₁
                (j, h) = first(stk)
                if cst′(σ[j], j′) ≥ cst′(σ[j], j′)
                    cst[σ[j]] = cst′(σ[j], j′)
                    spl[σ[j]] = j′
                    if h == j
                        pop!(stk)
                    end
                else
                    cst[j] = min(cst[j], cst′(j, j + 1))
                    spl[j] = j + 1
                    while !isempty(stk) && ((j′, h) = first(stk); (cst′(h, j + 1) < cst′(h, j′)))
                        pop!(stk)
                    end
                    if isempty(stk)
                        push!(stk, (j + 1, j₀))
                    else
                        (j′, h) = first(stk)
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
                        h′ = h′_lo

                        if h′ < j
                            push!(stk, (j + 1, h′))
                        end
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
=#