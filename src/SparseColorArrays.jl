struct VertexCount end

@inline (::VertexCount)(j, j′, k...) = Int(j′ - j)
@inline cost_type(::Type{VertexCount}) = Int
oracle_model(::VertexCount) = VertexCount()
oracle_stripe(::AbstractHint, ::VertexCount, ::SparseMatrixCSC; kwargs...) = VertexCount()
#bound_stripe(A::SparseMatrixCSC, K, ::VertexCount) = (size(A)[2]/K, size(A)[2]) $TODO ?

struct PinCount{Ti} <: AbstractMatrix{Ti}
    n::Int
    pos::Vector{Ti}
end

#@inline (::PinCount)(j, j′, k...) = Int(j′ - j)
#@inline cost_type(::Type{VertexCount}) = Int
#oracle_model(::VertexCount) = VertexCount()
#oracle_stripe(::AbstractHint, ::VertexCount, ::SparseMatrixCSC; kwargs...) = VertexCount()
#bound_stripe(A::SparseMatrixCSC, K, ::VertexCount) = (size(A)[2]/K, size(A)[2]) $TODO ?

Base.size(arg::PinCount) = (arg.n + 1, arg.n + 1)

pincount(args...; kwargs...) = pincount(NoHint(), args...; kwargs...)
pincount(::AbstractHint, args...; kwargs...) = @assert false
pincount(hint::AbstractHint, A::SparseMatrixCSC; kwargs...) =
    pincount!(hint, size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

pincount!(args...; kwargs...) = pincount!(NoHint(), args...; kwargs...)
pincount!(::AbstractHint, args...; kwargs...) = @assert false
pincount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}; kwargs...) where {Ti} = 
    PinCount(hint, n, pos; kwargs...)

PinCount(hint::AbstractHint, n, pos::Vector{Ti}; kwargs...) where {Ti} = 
    PinCount{Ti}(hint, n, pos; kwargs...)
function PinCount{Ti}(hint::AbstractHint, n, pos::Vector{Ti}; kwargs...) where {Ti}
    return PinCount(n, pos)
end

Base.getindex(arg::PinCount{Ti}, j::Integer, j′::Integer) where {Ti} = arg(j, j′)
function (arg::PinCount{Ti})(j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return arg.pos[j′] - arg.pos[j]
    end
end



struct NetCount{Ti, Lnk} <: AbstractMatrix{Ti}
    n::Int
    pos::Vector{Ti}
    lnk::Lnk
end

Base.size(arg::NetCount) = (arg.n + 1, arg.n + 1)

netcount(args...; kwargs...) = netcount(NoHint(), args...; kwargs...)
netcount(::AbstractHint, args...; kwargs...) = @assert false
netcount(hint::AbstractHint, A::SparseMatrixCSC; kwargs...) =
    netcount!(hint, size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

netcount!(args...; kwargs...) = netcount!(NoHint(), args...; kwargs...)
netcount!(::AbstractHint, args...; kwargs...) = @assert false
netcount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    NetCount(hint, m, n, N, pos, idx; kwargs...)

NetCount(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    NetCount{Ti}(hint, m, n, N, pos, idx; kwargs...)
function NetCount{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    @inbounds begin
        hst = zeros(Ti, m)
        idx′ = undefs(Ti, N)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                idx′[q] = (n + 1) - hst[i]
                hst[i] = j
            end
        end

        return NetCount(n, pos, dominancecount!(hint, n + 1, n + 1, N, pos, idx′; kwargs...))
    end
end

Base.getindex(arg::NetCount{Ti}, j::Integer, j′::Integer) where {Ti} = arg(j, j′)
function (arg::NetCount{Ti})(j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return (arg.pos[j′] - arg.pos[j]) - arg.lnk((arg.n + 2) - j, j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Same, _j′) where {Ti, Net <: NetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Same((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Next, _j′::Same) where {Ti, Net <: NetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Prev((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (ocl::Step{Net})(_j::Prev, _j′::Same) where {Ti, Net <: NetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = ocl.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Next((arg.n + 2) - j), _j′)
    end
end



struct SelfNetCount{Ti, Lnk} <: AbstractMatrix{Ti}
    n::Int
    lnk::Lnk
end

SelfNetCount{Ti}(n, lnk::Lnk) where {Ti, Lnk} = SelfNetCount{Ti, Lnk}(n, lnk)

Base.size(arg::SelfNetCount) = (arg.n + 1, arg.n + 1)

selfnetcount(args...; kwargs...) = selfnetcount(NoHint(), args...; kwargs...)
selfnetcount(::AbstractHint, args...; kwargs...) = @assert false
selfnetcount(hint::AbstractHint, A::SparseMatrixCSC; kwargs...) =
    selfnetcount!(hint, size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

selfnetcount!(args...; kwargs...) = selfnetcount!(NoHint(), args...; kwargs...)
selfnetcount!(::AbstractHint, args...; kwargs...) = @assert false
selfnetcount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SelfNetCount(hint, m, n, N, pos, idx; kwargs...)

SelfNetCount(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SelfNetCount{Ti}(hint, m, n, N, pos, idx; kwargs...)
function SelfNetCount{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    @inbounds begin
        hst = zeros(Ti, m)
        hst′ = undefs(Ti, m)
        idx′ = undefs(Ti, N)
        pos′ = zeros(Ti, n + 1)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                if hst[i] == 0
                    hst[i] = j
                end
                hst′[i] = j
            end
        end

        for i = 1:m
            if hst[i] != 0
                j′ = hst′[i]
                pos′[j′ + 1] += 1
            end
        end

        q = 1
        for j = 1:n + 1
            (pos′[j], q) = (q, q + pos′[j])
        end

        N′ = q - 1

        idx′ = undefs(Ti, N′)

        for i = 1:m
            if hst[i] != 0
                j = hst[i]
                j′ = hst′[i]
                q = pos′[j′ + 1]
                idx′[q] = (n + 1) - j
                pos′[j′ + 1] = q + 1
            end
        end

        return SelfNetCount{Ti}(n, dominancecount!(hint, n + 1, n + 1, N′, pos′, idx′; kwargs...))
    end
end

Base.getindex(arg::SelfNetCount{Ti}, j::Integer, j′::Integer) where {Ti} = arg(j, j′)
function (arg::SelfNetCount{Ti})(j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return arg.lnk((arg.n + 2) - j, j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Same, _j′) where {Ti, Net <: SelfNetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return Step(arg.lnk)(Same((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Next, _j′::Same) where {Ti, Net <: SelfNetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return Step(arg.lnk)(Prev((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (ocl::Step{Net})(_j::Prev, _j′::Same) where {Ti, Net <: SelfNetCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = ocl.ocl
        return Step(arg.lnk)(Next((arg.n + 2) - j), _j′)
    end
end



struct SelfPinCount{Ti, Lnk} <: AbstractMatrix{Ti}
    n::Int
    lnk::Lnk
end

SelfPinCount{Ti}(n, lnk::Lnk) where {Ti, Lnk} = SelfPinCount{Ti, Lnk}(n, lnk)

Base.size(arg::SelfPinCount) = (arg.n + 1, arg.n + 1)

selfpincount(args...; kwargs...) = selfpincount(NoHint(), args...; kwargs...)
selfpincount(::AbstractHint, args...; kwargs...) = @assert false
selfpincount(hint::AbstractHint, A::SparseMatrixCSC; kwargs...) =
    selfpincount!(hint, size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

selfpincount!(args...; kwargs...) = selfpincount!(NoHint(), args...; kwargs...)
selfpincount!(::AbstractHint, args...; kwargs...) = @assert false
selfpincount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SelfPinCount(hint, m, n, N, pos, idx; kwargs...)

SelfPinCount(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SelfPinCount{Ti}(hint, m, n, N, pos, idx; kwargs...)
function SelfPinCount{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    @inbounds begin
        pos′ = zeros(Ti, n + 1)
        idx′ = zeros(Ti, N)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                (i, i′) = minmax(j, i)
                pos′[i′ + 1] += 1
            end
        end

        q = 1
        for j = 1:n + 1
            (pos′[j], q) = (q, q + pos′[j])
        end

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                (i, i′) = minmax(j, i)
                q′ = pos′[i′ + 1]
                idx′[q′] = (n + 1) - i
                pos′[i′ + 1] = q′ + 1
            end
        end

        return SelfPinCount{Ti}(n, dominancecount!(hint, n + 1, n + 1, N, pos′, idx′; kwargs...))
    end
end

Base.getindex(arg::SelfPinCount{Ti}, j::Integer, j′::Integer) where {Ti} = arg(j, j′)
function (arg::SelfPinCount{Ti})(j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return arg.lnk((arg.n + 2) - j, j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Same, _j′) where {Ti, Net <: SelfPinCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return Step(arg.lnk)(Same((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Next, _j′::Same) where {Ti, Net <: SelfPinCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return Step(arg.lnk)(Prev((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (ocl::Step{Net})(_j::Prev, _j′::Same) where {Ti, Net <: SelfPinCount{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = ocl.ocl
        return Step(arg.lnk)(Next((arg.n + 2) - j), _j′)
    end
end