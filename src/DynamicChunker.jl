struct DynamicTotalChunker{F}
    f::F
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker, args...; kwargs...) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        f = oracle_stripe(method.f, A, args...)

        cst = Vector{cost_type(f)}(undef, n + 1)
        cst[1] = zero(cost_type(f))
        spl = Vector{Int}(undef, n + 1)
        for j′ = 2:n + 1
            best_c = cst[1] + f(1, j′)
            best_j = 1
            for j = 1 : j′ - 1
                c = cst[j] + f(j, j′) 
                if c < best_c
                    best_c = c
                    best_j = j
                end
            end
            cst[j′] = best_c
            spl[j′] = best_j
        end

        return unravel_chunks!(spl, n)
    end
end

function unravel_chunks!(spl, n)
    K = 0
    j′ = n + 1
    while j′ != 1
        j = spl[j′]
        spl[end - K] = j′
        K += 1
        j′ = j
    end
    spl[1] = 1
    for k = 1:K
        spl[k + 1] = spl[end - K + k]
    end
    resize!(spl, K + 1)
    return SplitPartition(K, spl)
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{<:ConstrainedCost}, args...; kwargs...) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        f = oracle_stripe(method.f.f, A, args...)
        w = oracle_stripe(method.f.w, A, args...)
        w_max = method.f.w_max

        cst = Vector{cost_type(f)}(undef, n + 1)
        cst[1] = zero(cost_type(f))
        spl = Vector{Int}(undef, n + 1)
        j₀ = 1
        for j′ = 2:n + 1
            while w(j₀, j′) > w_max
                j₀ += 1
            end
            @assert j₀ < j′ #TODO infeasibility

            best_c = cst[j₀] + f(j₀, j′)
            best_j = j₀
            for j = j₀ + 1 : j′ - 1
                c = cst[j] + f(j, j′) 
                if c < best_c
                    best_c = c
                    best_j = j
                end
            end
            cst[j′] = best_c
            spl[j′] = best_j
        end

        return unravel_chunks!(spl, n)
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
        f_ocl = oracle_stripe(f, A)

        Δ_net = zeros(Ti, n + 1) # Δ_net is the number of additional distinct entries we see as our part size grows.
        hst = zeros(Ti, m) # hst is the last time we saw some nonzero
        cst = Vector{cost_type(f)}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        spl = Vector{Ti}(undef, n + 1)
        if x_net isa Nothing
            x_net = Ref(Vector{Ti}(undef, n + 1)) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        else
            @assert x_net isa Ref{Vector{Int}}
            x_net[] = Vector{Ti}(undef, n + 1) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        end
        Δ_net[1] = 0
        cst[1] = zero(cost_type(f))
        d₀ = 0
        for j′ = 2:n + 1
            Δ_net[j′ - 1] = A_pos[j′] - A_pos[j′ - 1] # The number of distinct nonzero blocks in each candidate part
            for q = A_pos[j′ - 1] : A_pos[j′] - 1
                i = A_idx[q]
                j = hst[i]
                if j >= 1
                    Δ_net[j] -= 1
                else
                    d₀ += 1
                end
                hst[i] = j′ - 1
            end
            d = d₀
            best_c = cst[1] + f(1, A_pos[j′] - A_pos[1], d) #could be simpler
            best_d = d
            best_j = 1
            for j = 2 : j′ - 1
                d -= Δ_net[j - 1]
                c = cst[j] + f(j′ - j, A_pos[j′] - A_pos[j], d) 
                if c < best_c
                    best_c = c
                    best_d = d
                    best_j = j
                end
            end
            cst[j′] = best_c
            x_net[][j′] = best_d
            spl[j′] = best_j
        end

        K = 0
        j′ = n + 1
        while j′ != 1
            j = spl[j′]
            spl[end - K] = j′
            x_net[][end - K] = x_net[][j′]
            K += 1
            j′ = j
        end
        spl[1] = 1
        for k = 1:K
            spl[k + 1] = spl[end - K + k]
            x_net[][k + 1] = x_net[][end - K + k]
        end
        resize!(spl, K + 1)
        resize!(x_net[], K)
        return SplitPartition(K, spl)
    end
end


function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{<:ConstrainedCost{F}}, args...; x_net = nothing, kwargs...) where {F<:AbstractNetCostModel, Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        A_pos = A.colptr
        A_idx = A.rowval

        f = method.f.f
        w = oracle_stripe(method.f.w, A, args...)
        w_max = method.f.w_max


        Δ_net = zeros(Ti, n + 1) # Δ_net is the number of additional distinct entries we see as our part size grows.
        hst = zeros(Ti, m) # hst is the last time we saw some nonzero
        cst = Vector{cost_type(f)}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        spl = Vector{Ti}(undef, n + 1)
        if x_net isa Nothing
            x_net = Ref(Vector{Ti}(undef, n + 1)) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        else
            @assert x_net isa Ref{Vector{Int}}
            x_net[] = Vector{Ti}(undef, n + 1) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        end
        Δ_net[1] = 0
        cst[1] = zero(cost_type(f))
        d₀ = 0
        j₀ = 1
        for j′ = 2:n + 1
            while w(j₀, j′) > w_max #TODO infeasibiliity
                d₀ -= Δ_net[j₀]
                j₀ += 1
            end
            Δ_net[j′ - 1] = A_pos[j′] - A_pos[j′ - 1] # The number of distinct nonzero blocks in each candidate part
            for q = A_pos[j′ - 1] : A_pos[j′] - 1
                i = A_idx[q]
                j = hst[i]
                if j >= j₀
                    Δ_net[j] -= 1
                else 
                    d₀ += 1
                end
                hst[i] = j′ - 1
            end
            d = d₀
            best_c = cst[j₀] + f(1, A_pos[j′] - A_pos[j₀], d) #could be simpler
            best_d = d
            best_j = j₀
            for j = j₀ + 1 : j′ - 1
                d -= Δ_net[j - 1]
                c = cst[j] + f(j′ - j, A_pos[j′] - A_pos[j], d) 
                if c < best_c
                    best_c = c
                    best_d = d
                    best_j = j
                end
            end
            cst[j′] = best_c
            x_net[][j′] = best_d
            spl[j′] = best_j
        end

        K = 0
        j′ = n + 1
        while j′ != 1
            j = spl[j′]
            spl[end - K] = j′
            x_net[][end - K] = x_net[][j′]
            K += 1
            j′ = j
        end
        spl[1] = 1
        for k = 1:K
            spl[k + 1] = spl[end - K + k]
            x_net[][k + 1] = x_net[][end - K + k]
        end
        resize!(spl, K + 1)
        resize!(x_net[], K)
        return SplitPartition(K, spl)
    end
end