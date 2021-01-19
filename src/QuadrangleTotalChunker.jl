struct ConvexTotalChunker{F}
    f::F
end

#TODO add reference partitioners to test optimized constrained dynamic ones.
#TODO add optimized constrained dynamic partitioner before optimized convex one.

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::ConvexTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        ftr = Stack{Tuple{Ti, Ti}}(n)
        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[1] = zero(cost_type(f))
        f′(j, j′) = cst[j] + f(j, j′)
        chunk_convex!(cst, spl, f′, 1, n + 1, ftr)

        return unravel_chunks!(spl, n)
    end
end

#=
function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ConvexTotalChunker, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f, A, args...)

        if K == 1
            return SplitPartition{Ti}(1, [Ti(1), Ti(n + 1)])
        end

        ftr = Stack{Tuple{Ti, Ti}}(n)
        ptr = undefs(Ti, n + 1, K - 1)
        cst = fill(typemax(cost_type(f)), n + 2, K - 1)
        cst[n + 2, :] .= zero(cost_type(f))

        #TODO handle zero-length jumps through initialization, rather than indexing
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
=#

function chunk_convex!(cst, ptr, f, j₀, j′₁, ftr)
    empty!(ftr)
    push!(ftr, (j₀, j′₁))
    for j′ = j₀ + 1:j′₁
        (j, h) = first(ftr)
        if f(j′ - 1, j′) ≥ f(j, j′)
            if f(j, j′) <= cst[j′]
                cst[j′] = f(j, j′)
                ptr[j′] = j
            end
            if h == j′
                pop!(ftr)
            end
        else
            if f(j′ - 1, j′) <= cst[j′]
                cst[j′] = f(j′ - 1, j′)
                ptr[j′] = j′ - 1
            end
            while !isempty(ftr) && ((j, h) = first(ftr); (f(j′ - 1, h) < f(j, h)))
                pop!(ftr)
            end
            if isempty(ftr)
                push!(ftr, (j′ - 1, j′₁))
            else
                (j, h) = first(ftr)
                h_lo = j′
                h_hi = h
                #=
                h_ref = h_lo
                while h_ref < h_hi && f(j′ - 1, h_ref + 1) < f(j, h_ref + 1)
                    h_ref += 1
                end
                =#
                while h_lo <= h_hi
                    h = fld2(h_lo + h_hi)
                    if f(j′ - 1, h) < f(j, h)
                        h_lo = h + 1
                    else
                        h_hi = h - 1
                    end
                end
                h = h_hi

                if j′ < h
                    push!(ftr, (j′ - 1, h))
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
        σ_cst = undefs(cost_type(f), 2n + 2)

        spl = zeros(Ti, n + 1)
        cst = fill(typemax(cost_type(f)), n + 1)
        cst[1] = zero(cost_type(f))
        f′(j, j′) = cst[j] + f(j, j′)
        chunk_convex_constrained!(cst, spl, f′, w, w_max, 1, n + 1, ftr, σ_j, σ_j′, σ_cst, σ_ptr)

        return unravel_chunks!(spl, n)
    end
end

#=
function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::ConvexTotalChunker{<:ConstrainedCost}, args...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        f = oracle_stripe(method.f.f, A, args...)
        w = oracle_stripe(method.f.w, A, args...)
        w_max = method.f.w_max

        if K == 1
            return SplitPartition{Ti}(1, [Ti(1), Ti(n + 1)])
        end

        ftr = Stack{Tuple{Ti, Ti}}(2n)
        σ_j = undefs(Ti, 2n + 2)
        σ_j′ = undefs(Ti, 2n + 2)
        σ_ptr = undefs(Ti, 2n + 2)
        σ_cst = undefs(cost_type(method.f), 2n + 2)
        j_lo = undefs(Ti, K + 1)
        j′ = n + 1
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
        pos = undefs(Ti, K + 1)
        pos[1] = 1
        for k = 1:K
            if j′_hi[k] <= j_lo[k]
                spl = ones(Ti, K + 1)
                spl[end] = n + 1
                throw(ArgumentError("infeasible")) #TODO infeasibility
                return SplitPartition(K, spl)
            end
            pos[k + 1] = pos[k] + (1 + j′_hi[k] - j_lo[k])
        end
        ptr = undefs(Ti, pos[end] - 1)
        cst = fill(typemax(cost_type(f)), pos[end] - 1)

        for q = pos[K]:pos[K + 1] - 1
            j = j_lo[K] + (q - pos[K])
            cst[q] = f(j, n + 1)
            ptr[q] = n + 1
        end

        for k = K-1:-1:1
            cst[pos[k + 1] - 1] = zero(cost_type(f))
            #we need to set infeasible costs from level k + 1 to be inf and allocate space for those. Note that the cost is still convex.
            f′(i, i′) = cst[pos[k + 1] + i′ - 1] + f(j_lo[k] + i - 1, j_lo[k] + i′ - 2, k)
            chunk_convex_constrained!((@view cst[pos[k]:pos[k + 1] - 1]), (@view ptr[pos[k]:pos[k + 1] - 1]), f′, w, w_max, 1, j′_hi[k] - j_lo[k] + 2, ftr, σ_j, σ_j′, σ_cst, σ_ptr)
        end

        spl = undefs(Ti, K + 1)
        spl[1] = 1
        for k = 2:K
            spl[k] = ptr[pos[k - 1] + spl[k - 1] + j_lo[k - 1]] - 1
        end
        spl[K + 1] = n + 1
        return SplitPartition{Ti}(K, spl) 
    end
end


=#


function chunk_convex_constrained!(cst, ptr, f, w, w_max, J₀, J′₁, ftr, σ_j, σ_j′, σ_cst, σ_ptr)
    j′₁ = J₀ + 1
    while j′₁ < J′₁ && w(J₀, j′₁ + 1) <= w_max
        j′₁ += 1
    end
    j₀ = J₀

    while j′₁ <= J′₁
        chunk_convex!(cst, ptr, f, j₀, j′₁, ftr)

        if j′₁  == J′₁
            #@info "hello" j₀ J′₁
            break
        end
        j′ = j′₁
        I = 1
        for j = j₀ + 1 : j′₁ #j₀ is redundant, be we just deal with it, whatever.
            σ_j′[I] = j′
            I += 1
            σ_j[I] = j
            while j′ < J′₁ && w(j, j′ + 1) <= w_max
                j′ += 1
                σ_j′[I] = j′
                I += 1
                σ_j[I] = j
            end
        end
        #=
        while j′ < J′₁ && w(j′₁, j′ + 1) <= w_max
            j′ += 1
        end
        =#

        for i = 2:I-1
            σ_cst[i] = typemax(eltype(cst))
        end
        #@info "wut?" I σ_j[1:I-1] σ_j′[1:I-1] σ_cst[1:I - 1]

        #have i < i′
        #need if j = σ_j[I], then j′ < σ_j′[I] (can do this with i < i′)
        #also need j < j′ but this is satisfied for any i, i′ since rand(σ_j) < rand(σ_j′)

        σ_f(i, i′) = f(σ_j[I - i], σ_j′[I - i′])
        chunk_convex!(σ_cst, σ_ptr, σ_f, 1, I - 1, ftr)

        #@info "wut?" I σ_j[2:I - 1] σ_j′[1:I-2] σ_ptr[2:I - 1]
        for i′ = 2:I - 1
            #@info "wut?" σ_j[i] σ_j′[i] I σ_ptr[i] σ_cst[i]
            #@assert σ_j′[σ_ptr[i]] - σ_j[i] <= w_max
            if σ_j′[I - i′] > j′₁
                cst[σ_j′[I - i′]] = σ_cst[i′]
                ptr[σ_j′[I - i′]] = σ_j[I - σ_ptr[i′]]
            end
        end

        #=
        for j = σ_j[I - 1]:j₀
            @info "hmm" j₀ j₁ j ptr[j] w_max f′(j, ptr[j]) minimum(j′ -> j′ - j > w_max ? Inf : f′(j, j′), j₀:j₁) 
            @assert f′(j, ptr[j]) ≈ minimum(j′ -> j′ - j > w_max ? Inf : f′(j, j′), j₀:j₁) 
        end
        =#

        j₀ = j′₁
        j′₁ = σ_j′[I - 1]
        #@info "round4" j₀ j₁
    end
end