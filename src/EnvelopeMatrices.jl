struct EnvelopeMatrix{T} <: AbstractMatrix{Tuple{T, T}}
    H::Int
    m::Int
    n::Int
    tree::Vector{Tuple{T, T}}
end

Base.size(arg::EnvelopeMatrix) = (arg.n + 1, arg.n + 1)

function rowenvelope(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        H = cllog2(n)
        tree = undefs(Tuple{Ti, Ti}, (1 << (H + 1)) - 1)
        for j = 1 : n
            if A.colptr[j] < A.colptr[j + 1]
                tree[(1 << H) - 1 + j] = (A.rowval[A.colptr[j]], A.rowval[A.colptr[j + 1] - 1])
            else
                tree[(1 << H) - 1 + j] = (m + 1, 0)
            end
        end
        for j = (1 << H) + n : (1 << (H + 1)) - 1
            tree[j] = (m + 1, 0)
        end
        
        for j = (1 << (H + 1)) - 2:-2:1
            (left_lo, left_hi) = tree[j]
            (right_lo, right_hi) = tree[j + 1]
            tree[j >> 1] = (min(left_lo, right_lo), max(left_hi, right_hi))
        end
        return EnvelopeMatrix{Ti}(H, m, n, tree)
    end
end

function Base.getindex(arg::EnvelopeMatrix{T}, j, j′) where {T}
    @inbounds begin
        H = arg.H
        m = arg.m
        n = arg.n
        tree = arg.tree

        (res_lo, res_hi) = (T(m + 1), T(0))

        j = ((1 << H) - 1) + j
        j′ = ((1 << H) - 1) + (j′ - 1)

        while j <= j′
            (left_lo, left_hi) = tree[j]
            (right_lo, right_hi) = tree[j′]
            (res_lo, res_hi) = (min(res_lo, left_lo, right_lo), max(res_hi, left_hi, right_hi))
            j = (j + 1) >> 1
            j′ = (j′ - 1) >> 1
        end

        return (res_lo, res_hi)
    end
end