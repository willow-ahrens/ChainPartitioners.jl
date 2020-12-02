struct SparseCountedRowNet{Ti} <: AbstractMatrix{Ti}
    n::Int
    pos::Vector{Ti}
    lnk::SparseCountedArea{Ti}
end

Base.size(arg::SparseCountedRowNet) = (arg.n + 1, arg.n + 1)

struct SparseCountedLocalRowNet{Ti} <: AbstractArray{Ti, 3}
    n::Int
    m::Int
    N::Int
    K::Int
    prm::Vector{Int}
    Πos::Vector{Int}
    ΔΠos::Vector{Int}
    lnk::SparseCountedRooks{Ti}
end

Base.size(arg::SparseCountedLocalRowNet) = (arg.n + 1, arg.n + 1, arg.K)

struct SparseCountedLocalColNet{Ti} <: AbstractArray{Ti, 3}
    n::Int
    K::Int
    Πos::Vector{Ti}
    prm::Vector{Ti}
end

Base.size(arg::SparseCountedLocalColNet) = (arg.n + 1, arg.n + 1, arg.K)

rownetcount(A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti} =
    SparseCountedRowNet{Ti}(size(A)..., nnz(A), A.colptr, A.rowval; kwargs...)

SparseCountedRowNet(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    SparseCountedRowNet{Ti}(m, n, N, pos, idx; kwargs...)

function SparseCountedRowNet{Ti}(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
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

        return SparseCountedRowNet{Ti}(n, pos, SparseCountedArea{Ti}(n + 1, n + 1, N, pos, idx′; kwargs...))
    end
end

function Base.getindex(arg::SparseCountedRowNet{Ti}, j::Integer, j′::Integer) where {Ti}
    @inbounds begin
        return (arg.pos[j′] - arg.pos[j]) - arg.lnk[(arg.n + 2) - j, j′]
    end
end

localrownetcount(A::SparseMatrixCSC{Tv, Ti}, Π; kwargs...) where {Tv, Ti} =
    SparseCountedLocalRowNet{Ti}(size(A)..., nnz(A), Π.K, A.colptr, A.rowval, Π; kwargs...)

SparseCountedLocalRowNet(m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} = 
    SparseCountedLocalRowNet{Ti}(m, n, N, K, pos, idx, Π; kwargs...)

function SparseCountedLocalRowNet{Ti}(m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}; kwargs...) where {Ti}
    @inbounds begin
        hst = undefs(Ti, m)
        Πos = zeros(Ti, K + 1)
        ΔΠos = zeros(Ti, K + 1)
        prm = zeros(Ti, N)
        idx′ = zeros(Ti, N + m)

        for q in 1:N
            i = idx[q]
            k = Π.asg[i]
            Πos[k + 1] += 1
        end

        for i = 1:m
            k = Π.asg[i]
            Πos[k + 1] += 1
        end

        q = 1
        for k = 1:(K + 1)
            (Πos[k], q) = (q, q + Πos[k])
        end

        for i = 1:m
            k = Π.asg[i]
            p = Πos[k + 1]
            Δp = ΔΠos[k + 1]
            hst[i] = p + Δp
            ΔΠos[k + 1] = Δp + 1
        end

        for k = 1:K
            ΔΠos[k + 1] += ΔΠos[k]
        end

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                k = Π.asg[i]
                p = Πos[k + 1]
                Δp = ΔΠos[k + 1] - ΔΠos[k]
                prm[p - ΔΠos[k]] = j
                idx′[p] = (N + m + 1) - hst[i]
                hst[i] = p + Δp
                Πos[k + 1] = p + 1
            end
        end

        for i = 1:m
            k = Π.asg[i]
            p = Πos[k + 1]
            idx′[p] = (N + m + 1) - hst[i]
            Πos[k + 1] = p + 1
        end

        return SparseCountedLocalRowNet{Ti}(n, m, N, K, prm, Πos, ΔΠos, SparseCountedRooks{Ti}(N + m, idx′; kwargs...))
    end
end

function Base.getindex(arg::SparseCountedLocalRowNet{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = @view arg.prm[arg.Πos[k] - arg.ΔΠos[k] : arg.Πos[k + 1] - arg.ΔΠos[k + 1] - 1]
        rnk_j = arg.Πos[k] + searchsortedfirst(tmp, j) - 1
        rnk_j′ = arg.Πos[k] + searchsortedlast(tmp, j′ - 1)
        return (rnk_j′ - rnk_j) - arg.lnk[(arg.N + arg.m + 2) - (rnk_j + arg.ΔΠos[k + 1] - arg.ΔΠos[k]), rnk_j′]
    end
end

localcolnetcount(A::SparseMatrixCSC{Tv, Ti}, Π; kwargs...) where {Tv, Ti} =
    SparseCountedLocalColNet{Ti}(size(A)..., nnz(A), Π.K, A.colptr, A.rowval, Π; kwargs...)

SparseCountedLocalColNet(m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π; kwargs...) where {Ti} = 
    SparseCountedLocalColNet{Ti}(m, n, N, K, pos, idx, Π; kwargs...)

function SparseCountedLocalColNet{Ti}(m, n, N, K, pos::Vector{Ti}, idx::Vector{Ti}, Π::MapPartition{Ti}; kwargs...) where {Ti}
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

function Base.getindex(arg::SparseCountedLocalColNet{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = (@view arg.prm[arg.Πos[k] : arg.Πos[k + 1] - 1])
        rnk_j = searchsortedfirst(tmp, j) - 1
        rnk_j′ = searchsortedlast(tmp, j′ - 1)
        return rnk_j′ - rnk_j
    end
end