struct StrictChunker
    w_max::Int
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::StrictChunker, args...; kwargs...) where {Tv, Ti}
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        w_max = method.w_max

        A_pos = A.colptr
        A_idx = A.rowval

        #hst = zeros(Int, m)

        spl = Vector{Int}(undef, n + 1) # Column split locations

        c = A_pos[2] - A_pos[1] #The cardinality of the first column in the part
        j = 1
        K = 0
        spl[1] = 1
        for j′ = 2:n
            c′ = A_pos[j′ + 1] - A_pos[j′] #The cardinality of the candidate column
            w = j′ - j #Current block size
            d = true
            if c == c′ && w != w_max
                l′ = A_pos[j′]
                for l = A_pos[j]:(A_pos[j + 1] - 1)
                    if A_idx[l] != A_idx[l′]
                        d = false
                        break
                    end
                    l′ += 1
                end
            else
                d = false
            end
            if !d
                K += 1
                spl[K + 1] = j′
                j = j′
                c = c′
            end
        end
        K += 1
        spl[K + 1] = n + 1

        resize!(spl, K + 1)

        return SplitPartition{Ti}(K, spl)
    end
end