struct OverlapChunker
    ρ::Float64
    w_max::Int
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::OverlapChunker, args...; x_net = nothing) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        ρ = method.ρ
        w_max = method.w_max

        A_pos = A.colptr
        A_idx = A.rowval

        hst = zeros(Int, m)

        Π = Vector{Int}(undef, n + 1) # Column split locations
        if x_net isa Nothing
            x_net = Ref(Vector{Int}(undef, n + 1)) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        else
            @assert x_net isa Ref{Vector{Int}}
            x_net[] = Vector{Int}(undef, n + 1) # x_net[j] is the corresponding number of distinct nonzero entries in the part
        end

        d = A_pos[2] - A_pos[1] #The number of distinct values in the part
        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        K = 0
        Π[1] = 1
        for q in A_pos[1] : A_pos[2] - 1
            i = A_idx[q]
            hst[i] = 1
        end
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            d′ = d #Becomes the number of distinct values in the candidate part
            cc′ = 0 #The cardinality of the intersection between column j and j′
            for q = A_pos[j′] : A_pos[j′ + 1] - 1
                i = A_idx[q]
                h = hst[i]
                if abs(h) == j
                    cc′ += 1
                    hst[i] = -j′
                elseif j < h
                    hst[i] = j′
                elseif h < -j
                    cc′ += 1
                    hst[i] = -j′
                else
                    d′ += 1
                    hst[i] = j′
                end
            end
            w = j′ - j #Current block size
            if w == w_max || cc′ < ρ * min(c, c′)
                K += 1
                Π[K + 1] = j′
                x_net[][K] = d
                j = j′
                d = c′
            else
                d = d′
            end
        end
        j′ = n + 1
        w = j′ - j
        K += 1
        Π[K + 1] = j′
        resize!(Π, K + 1)
        resize!(x_net[], K)
        return SplitPartition{Ti}(K, Π)
    end
end