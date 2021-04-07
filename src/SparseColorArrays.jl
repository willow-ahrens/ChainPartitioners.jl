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

function (stp::Step{Net})(_j::Same, _j′) where {Ti, Net <: SparseCountedRowNet{Ti}}
    @inbounds begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Same((arg.n + 2) - j), _j′)
    end
end

function (stp::Step{Net})(_j::Next, _j′::Same) where {Ti, Net <: SparseCountedRowNet{Ti}}
    @inbounds begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = stp.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Prev((arg.n + 2) - j), _j′)
    end
end

function (ocl::Step{Net})(_j::Prev, _j′::Same) where {Ti, Net <: SparseCountedRowNet{Ti}}
    @inbounds begin
        j = destep(_j)
        j′ = destep(_j′)
        arg = ocl.ocl
        return (arg.pos[j′] - arg.pos[j]) - Step(arg.lnk)(Next((arg.n + 2) - j), _j′)
    end
end



@propagate_inbounds function localize(m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}) where {Ti}
    Πos = zeros(Ti, K + 1)
    πos = zeros(Ti, K + 1)
    idx′ = zeros(Ti, N)

    for j = 1:n
        k₀ = 0
        for q in pos[j] : pos[j + 1] - 1
            i = idx[q]
            k = Π.asg[i]
            πos[k + 1] += k != k₀
            k₀ = k
            Πos[k + 1] += 1
        end
    end

    q = 1
    j′ = 1
    for k = 1:(K + 1)
        (Πos[k], q) = (q, q + Πos[k])
        (πos[k], j′) = (j′, j′ + πos[k])
    end
    n′ = j′ - 1

    pos′ = undefs(Ti, n′ + 1)
    prm = zeros(Ti, n′)

    for j = 1:n
        k₀ = 0
        for q in pos[j] : pos[j + 1] - 1
            i = idx[q]
            k = Π.asg[i]
            q = Πos[k + 1]
            idx′[q] = i
            Πos[k + 1] = q + 1
            if k != k₀
                j′ = πos[k + 1]
                pos′[j′] = q
                prm[j′] = j
                πos[k + 1] = j′ + 1
            end
            k₀ = k
        end
    end
    pos′[n′ + 1] = N + 1
    return (n′, πos, prm, pos′, idx′)
end

#A = sparse([1, 2, 3, 4, 5, 6, 7, 8, 11, 2, 4, 5, 9, 10, 11, 1, 4, 6, 9, 10, 1, 5, 10, 7, 8, 3, 8, 7, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 17], [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, -0.05, -0.04, 1.0, 1.0, -0.05, -3.0, 0.5, 2.0, 0.6, 1.0, -1.0], 11, 17)

#println(A)
#println(localize(size(A)..., nnz(A), 3, A.colptr, A.rowval, MapPartition(3, [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])))
#exit()

struct SparseCountedLocalRowNet{Ti, Net} <: AbstractArray{Ti, 3}
    n::Int
    K::Int
    n′::Int
    πos::Vector{Ti}
    prm::Vector{Ti}
    net::Net
end

Base.size(arg::SparseCountedLocalRowNet) = (arg.n + 1, arg.n + 1, arg.K)

localrownetcount(args...; kwargs...) = localrownetcount(NoHint(), args...; kwargs...)
localrownetcount(::AbstractHint, args...; kwargs...) = @assert false
localrownetcount(hint::AbstractHint, A::SparseMatrixCSC, Π; kwargs...) =
    localrownetcount!(hint, size(A)..., nnz(A), Π.K, A.colptr, A.rowval, Π; kwargs...)

localrownetcount!(args...; kwargs...) = localrownetcount!(NoHint(), args...; kwargs...)
localrownetcount!(::AbstractHint, args...; kwargs...) = @assert false
localrownetcount!(hint::AbstractHint, m, n, N, K, pos, idx, Π; kwargs...) =
    SparseCountedLocalRowNet(hint, m, n, N, K, pos, idx, Π; kwargs...)

#You should avoid the searchsorted if you do the stepped oracle.
#The stepped interface should specify the previous input so that we don't need to track it in multiple places.
#Consider using hints in accesses (pushfirst, popfirst, push, pop, incstart, decstop, etc...) instead of complicated funky call.
SparseCountedLocalRowNet(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} = 
    SparseCountedLocalRowNet{Ti}(hint, m, n, N, K, pos, idx, Π; kwargs...)
function SparseCountedLocalRowNet{Ti}(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}; kwargs...) where {Ti}
    @inbounds begin
        (n′, πos, prm, pos′, idx′) = localize(m, n, N, K, pos, idx, Π)

        net = SparseCountedRowNet(hint, m, n′, N, pos′, idx′; kwargs...)

        return SparseCountedLocalRowNet{Ti, typeof(net)}(n, K, n′, πos, prm, net)
    end
end

Base.getindex(arg::SparseCountedLocalRowNet{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti} = arg(j, j′, k)
function (arg::SparseCountedLocalRowNet{Ti})(j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = @view arg.prm[arg.πos[k] : arg.πos[k + 1] - 1]
        rnk_j = arg.πos[k] + searchsortedfirst(tmp, j) - 1
        rnk_j′ = arg.πos[k] + searchsortedfirst(tmp, j′) - 1
        return arg.net(rnk_j, rnk_j′)
    end
end

mutable struct SteppedSparseCountedLocalRowNet{Ti, Net} <: AbstractArray{Ti, 3}
    rnk_j::Ti
    rnk_j′::Ti
    net::Net
end

Base.size(arg::SteppedSparseCountedLocalRowNet) = size(arg.net)

localrownetcount!(hint::StepHint, m, n, N, K, pos, idx, Π; kwargs...) =
    SteppedSparseCountedLocalRowNet(hint, m, n, N, K, pos, idx, Π; kwargs...)

SteppedSparseCountedLocalRowNet(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} = 
    SteppedSparseCountedLocalRowNet{Ti}(hint, m, n, N, K, pos, idx, Π; kwargs...)
function SteppedSparseCountedLocalRowNet{Ti}(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}; kwargs...) where {Ti}
    rnk_j = 0
    rnk_j′ = 0
    net = SparseCountedLocalRowNet{Ti}(hint, m, n, N, K, pos, idx, Π; kwargs...)
    return SteppedSparseCountedLocalRowNet(rnk_j, rnk_j′, net)
end

Base.getindex(arg::SteppedSparseCountedLocalRowNet{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti} = arg(j, j′, k)
function (arg::SteppedSparseCountedLocalRowNet{Ti})(j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = @view arg.net.prm[arg.net.πos[k] : arg.net.πos[k + 1] - 1]
        rnk_j = arg.net.πos[k] + searchsortedfirst(tmp, j) - 1
        arg.rnk_j = rnk_j
        rnk_j′ = arg.net.πos[k] + searchsortedfirst(tmp, j′) - 1
        arg.rnk_j′ = rnk_j′
        return arg.net.net(rnk_j, rnk_j′)
    end
end

function (stp::Step{Net})(_j::Same, _j′::Next, _k::Same) where {Ti, Net <: SteppedSparseCountedLocalRowNet{Ti}}
    @inbounds begin
        arg = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        rnk_j = arg.rnk_j
        rnk_j′ = arg.rnk_j′
        πos = arg.net.πos
        prm = arg.net.prm
        n′ = arg.net.n′

        if rnk_j′ < πos[k + 1] && prm[rnk_j′] < j′
            rnk_j′ += 1
            arg.rnk_j′ = rnk_j′
            return Step(arg.net.net)(Same(rnk_j), Next(rnk_j′))
        else
            return Step(arg.net.net)(Same(rnk_j), Same(rnk_j′))
        end
    end
end

function (stp::Step{Net})(_j::Next, _j′::Same, _k::Same) where {Ti, Net <: SteppedSparseCountedLocalRowNet{Ti}}
    @inbounds begin
        arg = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        rnk_j = arg.rnk_j
        rnk_j′ = arg.rnk_j′
        πos = arg.net.πos
        prm = arg.net.prm
        n′ = arg.net.n′

        if rnk_j < πos[k + 1] && prm[rnk_j] < j
            rnk_j += 1
            arg.rnk_j = rnk_j
            return Step(arg.net.net)(Next(rnk_j), Same(rnk_j′))
        else
            return Step(arg.net.net)(Same(rnk_j), Same(rnk_j′))
        end
    end
end

struct SparseCountedLocalColNet{Ti} <: AbstractArray{Ti, 3}
    n::Int
    K::Int
    Πos::Vector{Ti}
    prm::Vector{Ti}
end

Base.size(arg::SparseCountedLocalColNet) = (arg.n + 1, arg.n + 1, arg.K)

localcolnetcount(args...; kwargs...) = localcolnetcount(NoHint(), args...; kwargs...)
localcolnetcount(::AbstractHint, args...; kwargs...) = @assert false
localcolnetcount(hint::AbstractHint, A::SparseMatrixCSC{Tv, Ti}, Π; kwargs...) where {Tv, Ti} =
    localcolnetcount!(hint, size(A)..., nnz(A), Π.K, A.colptr, A.rowval, Π; kwargs...)

localcolnetcount!(args...; kwargs...) = localcolnetcount!(NoHint(), args...; kwargs...)
localcolnetcount!(::AbstractHint, args...; kwargs...) = @assert false
localcolnetcount!(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} =
    SparseCountedLocalColNet(hint, m, n, N, K, pos, idx, Π; kwargs...)

SparseCountedLocalColNet(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} = 
    SparseCountedLocalColNet{Ti}(hint, m, n, N, K, pos, idx, Π; kwargs...)
function SparseCountedLocalColNet{Ti}(hint::AbstractHint, m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}; kwargs...) where {Ti}
    @inbounds begin
        Πos = zeros(Ti, K + 1) 
        hst = zeros(Ti, K)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                k = Π.asg[i]
                if hst[k] != j
                    Πos[k + 1] += 1
                    hst[k] = j
                end
            end
        end
        
        q = 1
        for k = 1:(K + 1)
            (Πos[k], q) = (q, q + Πos[k])
        end

        prm = undefs(Ti, q)
        zero!(hst)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                k = Π.asg[i]
                if hst[k] != j
                    p = Πos[k + 1]
                    prm[p] = j
                    Πos[k + 1] = p + 1
                    hst[k] = j
                end
            end
        end

        return SparseCountedLocalColNet{Ti}(n, K, Πos, prm)
    end
end

Base.getindex(arg::SparseCountedLocalColNet{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti} = arg(j, j′, k)
function (arg::SparseCountedLocalColNet{Ti})(j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = (@view arg.prm[arg.Πos[k] : arg.Πos[k + 1] - 1])
        rnk_j = searchsortedfirst(tmp, j) - 1
        rnk_j′ = searchsortedlast(tmp, j′ - 1)
        return rnk_j′ - rnk_j
    end
end