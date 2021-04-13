struct SparseCountedRowNet{Ti, Lnk} <: AbstractMatrix{Ti}
    n::Int
    pos::Vector{Ti}
    lnk::Lnk
end

Base.size(arg::SparseCountedRowNet) = (arg.n + 1, arg.n + 1)

rownetcount(args...; kwargs...) = rownetcount(NoHint(), args...; kwargs...)
rownetcount(::AbstractHint, args...; kwargs...) = @assert false
rownetcount(hint::AbstractHint, A::SparseMatrixCSC; kwargs...) =
    rownetcount!(hint, size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

rownetcount!(args...; kwargs...) = rownetcount!(NoHint(), args...; kwargs...)
rownetcount!(::AbstractHint, args...; kwargs...) = @assert false
rownetcount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SparseCountedRowNet(hint, m, n, N, pos, idx; kwargs...)

SparseCountedRowNet(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SparseCountedRowNet{Ti}(hint, m, n, N, pos, idx; kwargs...)
function SparseCountedRowNet{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
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

        return SparseCountedRowNet(n, pos, areacount!(hint, n + 1, n + 1, N, pos, idx′; kwargs...))
    end
end

Base.getindex(arg::SparseCountedRowNet{Ti}, j::Integer, j′::Integer) where {Ti} = arg(j, j′)
function (arg::SparseCountedRowNet{Ti})(j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return (arg.pos[j′] - arg.pos[j]) - arg.lnk((arg.n + 2) - j, j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Same, _j′) where {Ti, Net <: SparseCountedRowNet{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Same((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Next, _j′::Same) where {Ti, Net <: SparseCountedRowNet{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Prev((arg.n + 2) - j), _j′)
    end
end

@propagate_inbounds function (ocl::Step{Net})(_j::Prev, _j′::Same) where {Ti, Net <: SparseCountedRowNet{Ti}}
    begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = ocl.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Next((arg.n + 2) - j), _j′)
    end
end
