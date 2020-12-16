struct DynamicTotalChunker{F}
    f::F
    w_max::Int
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{F}, args...; kwargs...) where {F, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        f = oracle_stripe(method.f, A, args...)
        w_max = method.w_max

        cst = Vector{cost_type(f)}(undef, n + 1)
        spl = Vector{Int}(undef, n + 1)
        for j = n:-1:1
            best_c = cst[j + 1] + f(j, j + 1)
            best_j′ = j + 1
            for j′ = j + 2 : min(j + w_max, n + 1)
                c = cst[j′] + f(j, j′) 
                if c < best_c
                    best_c = c
                    best_j′ = j′
                end
            end
            cst[j] = best_c
            spl[j] = best_j′
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

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{F}, args...; x_net = nothing, kwargs...) where {F<:AbstractNetCostModel, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        f = method.f
        w_max = method.w_max

        Δ_net = zeros(Int, n + 1) # Δ_net is the number of additional distinct entries we see as our part size grows.
        hst = fill(n + 1, m) # hst is the last time we saw some nonzero
        cst = Vector{cost_type(f)}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        spl = Vector{Int}(undef, n + 1)
        if x_net isa Nothing
            x_net = Ref(Vector{Int}(undef, n + 1)) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        else
            @assert x_net isa Ref{Vector{Int}}
            x_net[] = Vector{Int}(undef, n + 1) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        end
        Δ_net[n + 1] = 0
        cst[n + 1] = zero(cost_type(f))
        for j = n:-1:1
            d = A_pos[j + 1] - A_pos[j] # The number of distinct nonzero blocks in each candidate part
            Δ_net[j] = d
            for q = A_pos[j] : A_pos[j + 1] - 1
                i = A_idx[q]
                j′ = hst[i]
                if j′ <= j + w_max - 1
                    Δ_net[j′] -= 1
                end
                hst[i] = j
            end
            best_c = cst[j + 1] + f(1, d, d)
            best_d = d
            best_j′ = j + 1
            for j′ = j + 2 : min(j + w_max, n + 1)
                d += Δ_net[j′ - 1]
                c = cst[j′] + f(j′ - j, A_pos[j′] - A_pos[j], d) 
                if c < best_c
                    best_c = c
                    best_d = d
                    best_j′ = j′
                end
            end
            cst[j] = best_c
            x_net[][j] = best_d
            spl[j] = best_j′
        end

        K = 0
        j = 1
        while j != n + 1
            j′ = spl[j]
            K += 1
            spl[K] = j
            x_net[][K] = x_net[][j]
            j = j′
        end
        spl[K + 1] = j
        resize!(spl, K + 1)
        resize!(x_net[], K + 1)
        return SplitPartition{Ti}(K, spl)
    end
end