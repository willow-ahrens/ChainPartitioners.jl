struct ConcaveTotalChunker{F}
    f::F
end

struct ConcaveTotalSplitter{F}
    f::F
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConcaveTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(RandomHint(), method.f, A, args...)

        ftr = CircularDeque{Tuple{Ti, Ti}}(n + 1)
        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[1] = zero(cost_type(f))
        f′(j, j′) = cst[j] + f(j, j′)
        chunk_concave!(cst, spl, f′, 1, n + 1, ftr)

        return unravel_chunks!(spl, n)
    end
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ConcaveTotalSplitter, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(RandomHint(), method.f, A, args...)

        if K == 1
            return SplitPartition{Ti}(1, [Ti(1), Ti(n + 1)])
        end

        ftr = CircularDeque{Tuple{Ti, Ti}}(n + 1)

        ptr = zeros(Ti, n + 1, K)
        cst = fill(typemax(cost_type(f)), n + 1, K)

        for j′ = 1:n + 1
            cst[j′, 1] = f(1, j′, 1)
            ptr[j′, 1] = 1
        end
        for k = 2:K
            f′(j, j′) = cst[j, k - 1] + f(j, j′, k)
            for j′ = 1:n + 1
                cst[j′, k] = f′(j′, j′)
                ptr[j′, k] = j′ 
            end
            chunk_concave!((@view cst[:, k]), (@view ptr[:, k]), f′, 1, n + 1, ftr)
        end
        return unravel_splits(K, n, PermutedDimsArray(ptr, (2, 1)))
    end
end

#=
function chunk_concave!(cst, ptr, f::F, j₀, j′₁, ftr) where {F}
    @inbounds begin
        empty!(ftr)
        push!(ftr, (j₀, j₀ + 1))
        for j′ = j₀ + 1:j′₁
            (j, h) = first(ftr)
            c = f(j, j′)
            c′ = f(j′ - 1, j′)
            if c >= c′
                if c′ <= cst[j′]
                    cst[j′] = c′
                    ptr[j′] = j′ - 1
                end
                empty!(ftr)
                push!(ftr, (j′ - 1, j′ + 1))
            else
                if c <= cst[j′]
                    cst[j′] = c
                    ptr[j′] = j
                end
                while ((j, h) = last(ftr); (f(j′ - 1, h) <= f(j, h)))
                    pop!(ftr)
                end
                (j, h) = last(ftr)
                h_lo = h
                h_hi = j′₁

                h_ref = h_lo
                while h_ref < h_hi && f(j′ - 1, h_ref) > f(j, h_ref)
                    h_ref += 1
                end
                h = h_ref

                @assert h == h_hi || f(j′ - 1, h) <= f(j, h)
                @assert h <= min(j′ - 1, j) || f(j′ - 1, h - 1) > f(j, h - 1)
                #=
                while h_lo <= h_hi
                    h = fld2(h_lo + h_hi)
                    if f(j′ - 1, h) <= f(j, h)
                        h_hi = h - 1
                    else
                        h_lo = h + 1
                    end
                end
                h = h_hi
                =#
                @assert h > last(ftr)[2]
                push!(ftr, (j′ - 1, h))

                (j, h) = first(ftr)
                if j′ + 1 == h
                    popfirst!(ftr)
                else
                    (j, h) = popfirst!(ftr)
                    pushfirst!(ftr, (j, h + 1))
                end
            end
        end
    end
end

=#

function chunk_concave!(cst, ptr, f, j₀, j′₁, ftr)
    @inbounds begin
        for a = j₀:j′₁
            for b = a:j′₁
                for c = b + 1:j′₁
                    for d = c:j′₁
                        @assert f(a, c) + f(b, d) <= f(a, d) + f(b, c)
                    end
                end
            end
        end
    end
    
    @inbounds begin
        for j′ = j₀ + 1:j′₁
            for j = j₀:j′-1
                if f(j, j′) <= cst[j′]
                    cst[j′] = f(j, j′)
                    ptr[j′] = j
                end
            end
        end
    end
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ConcaveTotalSplitter{<:ConstrainedCost}, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(RandomHint(), method.f.f, A, args...)
        w = oracle_stripe(StepHint(), method.f.w, A, args...)
        w_max = method.f.w_max

        ftr = CircularDeque{Tuple{Ti, Ti}}(n + 1)

        (j′_lo, j′_hi) = column_constraints(A, K, w, w_max)

        if j′_hi[K] < n + 1
            spl = ones(Ti, K + 1)
            spl[end] = n + 1
            #TODO throw(ArgumentError("infeasible"))
            return SplitPartition(K, spl)
        end

        ptr = WindowConstrainedMatrix{Ti, Ti}(zero(Ti), n + 1, K, j′_lo, j′_hi)
        cst = WindowConstrainedMatrix{extend(cost_type(f)), Ti}(infinity(cost_type(f)), n + 1, K, j′_lo, j′_hi, ptr.pos)

        for j′ = j′_lo[1] : j′_hi[1]
            cst[j′, 1] = extend(f(1, j′, 1))
            ptr[j′, 1] = 1
        end

        for k = 2:K
            f′(j, j′) = cst[j, k - 1] + extend(f(j, j′, k))
            for j′ = j′_lo[k]:j′_hi[k]
                cst[j′, k] = f′(j′, j′)
                ptr[j′, k] = j′ 
            end
            chunk_concave!((@view cst[:, k]), (@view ptr[:, k]), f′, j′_lo[k - 1], j′_hi[k], ftr)
        end

        return unravel_splits(K, n, PermutedDimsArray(ptr, (2, 1)))
    end
end