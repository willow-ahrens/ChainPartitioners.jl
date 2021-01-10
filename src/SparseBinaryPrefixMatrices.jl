struct SparseBinaryCountedArea{Ti, Tb} <: AbstractMatrix{Ti}
    m::Int
    n::Int
    N::Int
    H::Int
    pos::Vector{Ti}
    qos::Vector{Ti}
    byt::Array{Tb, 2}
    cnt::Array{Ti, 2}
end

Base.size(arg::SparseBinaryCountedArea) = (arg.m + 1, arg.n + 1)

struct SparseBinaryCountedRooks{Ti, Tb} <: AbstractMatrix{Ti}
    N::Int
    b′::Int
    H::Int
    byt::Array{Tb, 2}
    cnt::Array{Ti, 2}
end

Base.size(arg::SparseBinaryCountedRooks) = (arg.N + 1, arg.N + 1)

function areacountbinary(A::SparseMatrixCSC{Tv, Ti}; Tb = UInt8, kwargs...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        return SparseBinaryCountedArea{Ti, Tb}(m, n, nnz(A), copy(A.colptr), copy(A.rowval); kwargs...)
    end
end

SparseBinaryCountedArea(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = SparseBinaryCountedArea{Ti, Int}(m, n, N, pos, idx; kwargs...)

function SparseBinaryCountedArea{Ti, Tb}(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}) where {Ti, Tb}
    #@inbounds begin
    begin
        #b = branching factor of tree = 1

        #H = height of tree
        H = cllog2(m + 1)

        qos = zeros(Ti, m + 2)
        qos[1] = 1
        qos[end] = N + 1
        cnt = zeros(Ti, 1 + cld(N, nbits(Tb)), H) #cnt = cached cumulative counts
        
        byt = zeros(Tb, 1 + cld(N, nbits(Tb)), H)
        idx′ = undefs(Ti, N)

        for h = H : -1 : 1 #h = level in tree
            cnt_2 = 0
            for i′ = 1 : 1 << h : m + 1 #i′ = "quotient" of sorts
                bkt_1 = 0
                bkt_2 = 0
                for q = qos[i′] : qos[min(i′ + (1 << h), end)] - 1 #q = position in current level
                    i = idx[q]
                    d = i >> (h - 1) & 1 #d = a "remainder" of sorts
                    bkt_2 += 1 - d
                end
                bkt_1 = qos[i′]
                bkt_2 += bkt_1
                for q = qos[i′] : qos[min(i′ + (1 << h), end)] - 1
                    i = idx[q]
                    d = i >> (h - 1) & 1
                    q′ = ifelse(d == 0, bkt_1, bkt_2)
                    idx′[q′] = i
                    Q = ((q - 1) >> log2nbits(Tb)) + 1
                    byt[Q, h] |= d << ((q - 1) & (nbits(Tb) - 1))
                    cnt[Q + 1, h] = cnt_2 += d
                    bkt_1 += 1 - d
                    bkt_2 += d
                end
                qos[min(i′ + 1 << (h - 1), end)] = bkt_1
                qos[min(i′ + 2 << (h - 1), end)] = bkt_2 #redundant I think?
            end

            #@info h idx cnt[:, h] qos

            idx, idx′ = idx′, idx
        end
        return SparseBinaryCountedArea{Ti, Tb}(m, n, N, H, pos, qos, byt, cnt)
    end
end

#we have removed b′ because increasing it doesn't significantly reduce the storage cost or preprocessing cost of the structure.

function Base.getindex(arg::SparseBinaryCountedArea{Ti, Tb}, i::Integer, j::Integer) where {Ti, Tb}
    #@inbounds begin
    begin
        H = arg.H
        pos = arg.pos
        qos = arg.qos
        byt = arg.byt
        cnt = arg.cnt

        Δq = pos[j] - 1
        i = Ti(i - 1)::Ti
        s = Ti(0)
        for h = H : -1 : 1
            i′ = i & ~(1 << h - 1) + 1
            q₁ = qos[i′] - 1
            q₂ = q₁ + Δq
            d = i >> (h - 1) & Ti(1)

            Q₁ = (q₁ >> log2nbits(Tb)) + 1
            Q₂ = (q₂ >> log2nbits(Tb)) + 1
            bkt_2 = cnt[Q₂, h] - cnt[Q₁, h]
            if h == H
                @assert bkt_2 == sum(count_ones.(byt[Q₁:(Q₂ - 1), h])) "$bkt_2 == $(sum(count_ones.(byt[Q₁:(Q₂ - 1), h])))"
            end
            bkt_2 += count_ones(byt[Q₂, h] & ((one(Tb) << (q₂ & (nbits(Tb) - 1))) - 1))
            bkt_2 -= count_ones(byt[Q₁, h] & ((one(Tb) << (q₁ & (nbits(Tb) - 1))) - 1))
            bkt_1 = Δq - bkt_2
            s += ifelse(d == 0, 0, bkt_1)
            Δq = ifelse(d == 0, bkt_1, bkt_2)
        end

        return s + Δq
    end
end