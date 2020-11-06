using Base: @pure, @propagate_inbounds

@inline function fllog2(x::T) where {T <: Integer}
    return (sizeof(T) * 8 - 1) - leading_zeros(x)
end
@inline function cllog2(x::T) where {T <: Integer}
    return fllog2(x - 1) + 1
end

@inline function fld2(x::T) where {T <: Integer}
    return x >>> true
end
@inline function cld2(x::T) where {T <: Integer}
    return (x + true) >>> true
end

@pure nbits(::Type{T}) where {T} = sizeof(T) * 8
@pure log2nbits(::Type{T}) where {T} = flrlog2(nbits(T))

@inline function flpow1m(x::T) where {T <: Integer}
    return (1 << x) - 1
end

@inline undefs(T::Type, dims::Vararg{Any, N}) where {N} = Array{T, N}(undef, dims...)

zero!(arr) = fill!(arr, zero(eltype(arr)))

function pattern(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return SparseMatrixCSC{Bool, Ti}(size(A)..., A.colptr, A.rowval, ones(Bool, nnz(A)))
end

function adjointpattern(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        pos = A.colptr
        idx = A.rowval
        pos′ = zeros(Ti, m + 1)
        idx′ = undefs(Ti, N)
        for j = 1:n
            for q = pos[j]:pos[j + 1]-1
                i = idx[q]
                pos′[i+1] += 1
            end
        end
        tmp = 1
        for i = 1:m + 1
            (pos′[i], tmp) = (tmp, tmp + pos′[i])
        end
        for j = 1:n
            for q = pos[j]:pos[j + 1]-1
                i = idx[q]
                q′ = pos′[i + 1]
                idx′[q′] = j
                pos′[i + 1] = q′ + 1
            end
        end
        return SparseMatrixCSC(n, m, pos′, idx′, A.nzval)
    end
end