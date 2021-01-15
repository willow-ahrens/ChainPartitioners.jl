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

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ConvexTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        if K == 1
            return SplitPartition{Ti}(1, [Ti(1), Ti(n + 1)])
        end

        ftr = Stack{Tuple{Ti, Ti}}(n)
        ptr = undefs(Ti, n + 1, K - 1)
        ptr[n + 1, :] .= n + 2
        cst = fill(typemax(cost_type(f)), n + 2, K - 1)
        cst[n + 2, :] .= zero(cost_type(f))

        f′₀(j, j′) = f(j, j′ - 1, K - 1) + f(j′ - 1, n + 1, K)
        chunk_convex!((@view cst[:, K - 1]), (@view ptr[:, K - 1]), f′₀, 1, n + 2, ftr)
        for k = K-2:-1:1
            f′(j, j′) = cst[j′, k] + f(j, j′ - 1, k)
            chunk_convex!((@view cst[:, k]), (@view ptr[:, k]), f′, 1, n + 2, ftr)
        end

        spl = undefs(Ti, K + 1)
        spl[1] = 1
        for k = 2:K
            spl[k] = ptr[spl[k - 1], k - 1] - 1
        end
        spl[K + 1] = n + 1
        return SplitPartition{Ti}(K, spl) 
    end
end

function chunk_convex!(cst, ptr, f′, j₀, j₁, ftr)
    empty!(ftr)
    push!(ftr, (j₁, j₀))
    for j = j₁ - 1:-1:j₀
        (j′, h) = first(ftr)
        if f′(j, j + 1) ≥ f′(j, j′)
            if f′(j, j′) <= cst[j]
                cst[j] = f′(j, j′)
                ptr[j] = j′
            end
            if h == j
                pop!(ftr)
            end
        else
            if f′(j, j + 1) <= cst[j]
                cst[j] = f′(j, j + 1)
                ptr[j] = j + 1
            end
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

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConvexTotalChunker{<:ConstrainedCost}, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f.f, A, args...)
        w = oracle_stripe(method.f.w, A, args...)
        w_max = method.f.w_max

        ftr = Stack{Tuple{Ti, Ti}}(2n)
        σ_j = undefs(Ti, 2n + 2)
        σ_j′ = undefs(Ti, 2n + 2)
        σ_ptr = undefs(Ti, 2n + 2)
        σ_cst = undefs(cost_type(method.f), 2n + 2)

        spl = zeros(Ti, n + 1)
        cst = fill(Inf, n + 1)
        cst[n + 1] = zero(cost_type(f))
        f′(j, j′) = cst[j′] + f(j, j′)
        chunk_convex_constrained!(cst, spl, f′, w, w_max, 1, n + 1, ftr, σ_j, σ_j′, σ_cst, σ_ptr)

        @info "funky"
        @info spl
        @info cst

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

function chunk_convex_constrained!(cst, ptr, f′, w, w_max, J₀, J₁, ftr, σ_j, σ_j′, σ_cst, σ_ptr)
    j₀ = J₁
    while j₀ > J₀ && w(j₀ - 1, J₁) <= w_max
        j₀ -= 1
    end
    j₁ = J₁

    while j₀ >= J₀
        chunk_convex!(cst, ptr, f′, j₀, j₁, ftr)

        if j₀ == J₀
            break
        end
        j = j₀
        I = 1
        for j′ = j₁-1:-1:j₀
            σ_j[I] = j
            σ_j′[I] = j′
            I += 1
            while j > J₀ && w(j - 1, j′) <= w_max
                j -= 1
                σ_j[I] = j
                σ_j′[I] = j′
                I += 1
            end
        end

        for i = 1:I
            σ_cst[i] = Inf
        end
        σ_f′(i, i′) = f′(σ_j[i], σ_j′[i′])
        chunk_convex!(σ_cst, σ_ptr, σ_f′, 1, I - 1, ftr)

        for i = I - 2:-1:1
            if σ_j[i] < j₀
                cst[σ_j[i]] = σ_cst[i]
                ptr[σ_j[i]] = σ_j′[σ_ptr[i]]
            end
        end
        j₁ = j₀
        j₀ = σ_j[I - 1]
        #@info "round4" j₀ j₁
    end
end