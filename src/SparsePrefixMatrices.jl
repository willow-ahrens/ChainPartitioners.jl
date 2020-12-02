struct SparseSummedArea{Tv, Ti} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    N::Int
    b::Int
    b′::Int
    H::Int
    pos::Vector{Ti}
    qos::Vector{Ti}
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
    wgt::Array{Tv, 2}
    scn::Array{Tv, 3}
end

Base.size(arg::SparseSummedArea) = (arg.m + 1, arg.n + 1)

struct SparseCountedArea{Ti} <: AbstractMatrix{Ti}
    m::Int
    n::Int
    N::Int
    b::Int
    b′::Int
    H::Int
    pos::Vector{Ti}
    qos::Vector{Ti}
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
end

Base.size(arg::SparseCountedArea) = (arg.m + 1, arg.n + 1)

struct SparseSummedRooks{Ti, Tv} <: AbstractMatrix{Tv}
    N::Int
    b::Int
    b′::Int
    H::Int
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
    wgt::Array{Tv, 2}
    scn::Array{Tv, 3}
end

Base.size(arg::SparseSummedRooks) = (arg.N + 1, arg.N + 1)

struct SparseCountedRooks{Ti} <: AbstractMatrix{Ti}
    N::Int
    b::Int
    b′::Int
    H::Int
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
end

Base.size(arg::SparseCountedRooks) = (arg.N + 1, arg.N + 1)

function areasum(A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    (m, n) = size(A)
    N = nnz(A)
    pos = copy(A.colptr)
    idx = copy(A.rowval)
    val = A.nzval
    return SparseSummedArea{Tv, Ti}(m, n, N, pos, idx, val; kwargs...)
end

SparseSummedArea(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Tv, Ti} = SparseSummedArea{Tv, Ti}(m, n, N, pos, idx, val; kwargs...)

function SparseSummedArea{Tv, Ti}(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; b = nothing, H = nothing, b′ = nothing) where {Tv, Ti}
    @inbounds begin
        if b === nothing #b = branching factor of tree
            if H === nothing
                b = cld(cllog2(m + 1), 3)
            else
                b = cld(cllog2(m + 1), H)
            end
        end

        if H === nothing #H = height of tree
            H = cld(cllog2(m + 1), b)
        end

        if b′ === nothing #b′ = cache period
            b′ = b + cllog2(H)
        end

        qos = zeros(Ti, m + 2)
        qos[1] = 1
        qos[end] = N + 1
        bkt = undefs(Ti, 1 << b + 1) #bkt = scratch space for cumulative count
        cnt = undefs(Ti, 1 << b, N >> b′ + 1, H - 1) #cnt = cached cumulative counts
        wgt = undefs(Tv, N, H) #wgt = permuted copies of weights
        for q = 1:N
            wgt[q, end] = val[q]
        end
        pre = undefs(Tv, 1 << b + 1) #pre = scratch space for cumulative sum
        scn = undefs(Tv, 1 << b + 1, N >> b′ + 1, H) #scn = cached cumulative sums
        
        byt = undefs(Ti, N)

        for h = H : -1 : 2 #h = level in tree
            for i′ = 1 : 1 << (h * b) : m + 1 #i′ = "quotient" of sorts
                zero!(bkt)
                for q = qos[i′] : qos[min(i′ + (1 << (h * b)), end)] - 1 #q = position in current level
                    i = idx[q]
                    d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                    bkt[d + 1] += 1
                end
                bkt[1] = qos[i′]
                for d = 1:(1 << b)
                    bkt[d + 1] = bkt[d] + bkt[d + 1]
                end
                for q = qos[i′] : qos[min(i′ + (1 << (h * b)), end)] - 1
                    i = idx[q]
                    d = i >> ((h - 1) * b) & (1 << b - 1) + 1
                    q′ = bkt[d]
                    byt[q′] = (idx[q′] & ~(1 << ((h - 1) * b) - 1)) | (i & (1 << ((h - 1) * b) - 1))
                    wgt[q′, h - 1] = wgt[q, h]
                    bkt[d] = q′ + 1
                end
                for d = 1 : 1 << b
                    qos[min(i′ + d << ((h - 1) * b), end)] = bkt[d]
                end
            end
            for d = 1 : 1 << b
                cnt[d, 1, h - 1] = 0
            end
            scn[1, 1, h] = 0
            for d = 1 : 1 << b
                scn[d + 1, 1, h] = 0
            end
            zero!(bkt)
            zero!(pre)
            for q = 1 : N
                i = idx[q]
                d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                bkt[d] += 1
                pre[d] += wgt[q, h]
                if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                    Q = (q >> b′) + 1
                    scn[1, Q, h] = 0
                    for d = 1 : 1 << b
                        cnt[d, Q, h - 1] = bkt[d]
                        scn[d + 1, Q, h] = pre[d] + scn[d, Q, h]
                    end
                end
            end

            idx, byt = byt, idx
        end

        for i′ = 1 : 1 << b : m + 1 #i′ = "quotient" of sorts
            zero!(bkt)
            for q = qos[i′] : qos[min(i′ + (1 << b), end)] - 1 #q = position in current level
                i = idx[q]
                d = i & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                bkt[d + 1] += 1
            end
            bkt[1] = qos[i′]
            for d = 1:(1 << b)
                bkt[d + 1] = bkt[d] + bkt[d + 1]
            end
            for q = qos[i′] : qos[min(i′ + (1 << b), end)] - 1
                i = idx[q]
                d = i & (1 << b - 1) + 1
                bkt[d] += 1
            end
            for d = 1 : 1 << b
                qos[min(i′ + d, end)] = bkt[d]
            end
        end
        scn[1, 1, 1] = 0
        for d = 1 : 1 << b
            scn[d + 1, 1, 1] = 0
        end
        zero!(pre)
        for q = 1 : N
            i = idx[q]
            d = i & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
            pre[d] += wgt[q, 1]
            if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                Q = (q >> b′) + 1
                scn[1, Q, 1] = 0
                for d = 1 : 1 << b
                    scn[d + 1, Q, 1] = pre[d] + scn[d, Q, 1]
                end
            end
        end

        byt = idx

        return SparseSummedArea{Tv, Ti}(m, n, N, b, b′, H, pos, qos, byt, cnt, wgt, scn)
    end
end

function Base.getindex(arg::SparseSummedArea{Tv, Ti}, i::Integer, j::Integer) where {Tv, Ti}
    @inbounds begin
        b = arg.b
        b′ = arg.b′
        H = arg.H
        pos = arg.pos
        qos = arg.qos
        byt = arg.byt
        cnt = arg.cnt
        wgt = arg.wgt
        scn = arg.scn

        Δq = pos[j] - 1
        i = Ti(i - 1)::Ti
        s = Tv(0)
        for h = H : -1 : 2
            i′ = i & ~(1 << (h * b) - 1) + 1
            q₁ = qos[i′] - 1
            q₂ = q₁ + Δq
            d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1
            Q₁ = (q₁ >> b′) + 1
            Q₂ = (q₂ >> b′) + 1
            s += scn[d, Q₂, h] - scn[d, Q₁, h]
            Δq = cnt[d, Q₂, h - 1] - cnt[d, Q₁, h - 1]
            for q = (Q₁ - 1) << b′ + 1 : q₁
                d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                if d′ < d
                    s -= wgt[q, h]
                end
                Δq -= d′ == d
            end
            for q = (Q₂ - 1) << b′ + 1 : q₂
                d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                if d′ < d
                    s += wgt[q, h]
                end
                Δq += d′ == d
            end
        end
        i′ = i & ~(1 << b - 1) + 1
        q₁ = qos[i′] - 1
        q₂ = q₁ + Δq

        d = i & ((1 << b) - 1) + 1
        Q₁ = (q₁ >> b′) + 1
        Q₂ = (q₂ >> b′) + 1
        s += scn[d + 1, Q₂, 1] - scn[d + 1, Q₁, 1]
        for q = (Q₁ - 1) << b′ + 1 : q₁
            d′ = byt[q] & ((1 << b) - 1) + 1
            if d′ <= d
                s -= wgt[q, 1]
            end
        end
        for q = (Q₂ - 1) << b′ + 1 : q₂
            d′ = byt[q] & ((1 << b) - 1) + 1
            if d′ <= d
                s += wgt[q, 1]
            end
        end

        return s
    end
end

function areacount(A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        return SparseCountedArea{Ti}(m, n, nnz(A), copy(A.colptr), copy(A.rowval); kwargs...)
    end
end

SparseCountedArea(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = SparseCountedArea(m, n, N, pos, idx; kwargs...)

function SparseCountedArea{Ti}(m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; b = nothing, H = nothing, b′ = nothing) where {Ti}
    @inbounds begin
        if b === nothing #b = branching factor of tree
            if H === nothing
                b = cld(cllog2(m + 1), 3)
            else
                b = cld(cllog2(m + 1), H)
            end
        end

        if H === nothing #H = height of tree
            H = cld(cllog2(m + 1), b)
        end

        if b′ === nothing #b′ = cache period
            b′ = b + cllog2(H)
        end

        qos = zeros(Ti, m + 2)
        qos[1] = 1
        qos[end] = N + 1
        bkt = undefs(Ti, 1 << b + 1) #bkt = scratch space for cumulative count
        cnt = undefs(Ti, 1 << b + 1, N >> b′ + 1, H) #cnt = cached cumulative counts
        
        byt = undefs(Ti, N)

        for h = H : -1 : 1 #h = level in tree
            for i′ = 1 : 1 << (h * b) : m + 1 #i′ = "quotient" of sorts
                zero!(bkt)
                for q = qos[i′] : qos[min(i′ + (1 << (h * b)), end)] - 1 #q = position in current level
                    i = idx[q]
                    d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                    bkt[d + 1] += 1
                end
                bkt[1] = qos[i′]
                for d = 1:(1 << b)
                    bkt[d + 1] = bkt[d] + bkt[d + 1]
                end
                for q = qos[i′] : qos[min(i′ + (1 << (h * b)), end)] - 1
                    i = idx[q]
                    d = i >> ((h - 1) * b) & (1 << b - 1) + 1
                    q′ = bkt[d]
                    byt[q′] = (idx[q′] & ~(1 << ((h - 1) * b) - 1)) | (i & (1 << ((h - 1) * b) - 1))
                    bkt[d] = q′ + 1
                end
                for d = 1 : 1 << b
                    qos[min(i′ + d << ((h - 1) * b), end)] = bkt[d]
                end
            end
            cnt[1, 1, h] = 0
            for d = 1 : 1 << b
                cnt[d + 1, 1, h] = 0
            end
            zero!(bkt)
            for q = 1 : N
                i = idx[q]
                d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                bkt[d] += 1
                if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                    Q = (q >> b′) + 1
                    cnt[1, Q, h] = 0
                    for d = 1 : 1 << b
                        cnt[d + 1, Q, h] = bkt[d] + cnt[d, Q, h]
                    end
                end
            end

            idx, byt = byt, idx
        end
        byt = idx
        return SparseCountedArea{Ti}(m, n, N, b, b′, H, pos, qos, byt, cnt)
    end
end

function Base.getindex(arg::SparseCountedArea{Ti}, i::Integer, j::Integer) where {Ti}
    @inbounds begin
        b = arg.b
        b′ = arg.b′
        H = arg.H
        pos = arg.pos
        qos = arg.qos
        byt = arg.byt
        cnt = arg.cnt

        Δq = pos[j] - 1
        i = Ti(i - 1)::Ti
        s = Ti(0)
        for h = H : -1 : 2
            i′ = i & ~(1 << (h * b) - 1) + 1
            q₁ = qos[i′] - 1
            q₂ = q₁ + Δq
            d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1
            Q₁ = (q₁ >> b′) + 1
            Q₂ = (q₂ >> b′) + 1
            s += cnt[d, Q₂, h] - cnt[d, Q₁, h]
            Δq = (cnt[d + 1, Q₂, h] - cnt[d, Q₂, h]) - (cnt[d + 1, Q₁, h] - cnt[d, Q₁, h])
            msk = ((1 << b) - 1) << ((h - 1) * b)
            cmp = (d - 1) << ((h - 1) * b)
            for q = (Q₁ - 1) << b′ + 1 : q₁
                d′ = byt[q] & msk
                s -= d′ < cmp
                Δq -= d′ == cmp
                #d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                #s -= d′ < d
                #Δq -= d′ == d
            end
            for q = (Q₂ - 1) << b′ + 1 : q₂
                d′ = byt[q] & msk
                s += d′ < cmp
                Δq += d′ == cmp
                #d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                #s += d′ < d
                #Δq += d′ == d
            end
        end

        i′ = i & ~(1 << b - 1) + 1
        q₁ = qos[i′] - 1
        q₂ = q₁ + Δq

        d = i & ((1 << b) - 1) + 1
        Q₁ = (q₁ >> b′) + 1
        Q₂ = (q₂ >> b′) + 1
        s += cnt[d + 1, Q₂, 1] - cnt[d + 1, Q₁, 1]
        msk = (1 << b) - 1
        cmp = d - 1
        for q = (Q₁ - 1) << b′ + 1 : q₁
            d′ = byt[q] & msk
            s -= d′ <= cmp
            #d′ = byt[q] & ((1 << b) - 1) + 1
            #s -= d′ <= d
        end
        for q = (Q₂ - 1) << b′ + 1 : q₂
            d′ = byt[q] & msk
            s += d′ <= cmp
            #d′ = byt[q] & ((1 << b) - 1) + 1
            #s += d′ <= d
        end

        return s
    end
end

SparseSummedRooks(N, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Ti, Tv} = SparseSummedRooks{Ti, Tv}(N, idx, val; kwargs...)

function SparseSummedRooks{Ti, Tv}(N, idx::Vector{Ti}, val::Vector{Tv}; b = nothing, H = nothing, b′ = nothing) where {Ti, Tv}
    @inbounds begin
        @debug begin
            @assert length(idx) == N
            hash = falses(N)
            for i in idx
                @assert 1 <= i <= N
                @assert !hash[i]
                hash[i] = true
            end
        end

        if b === nothing #b = branching factor of tree
            if H === nothing
                b = cld(cllog2(N + 1), 3)
            else
                b = cld(cllog2(N + 1), H)
            end
        end

        if H === nothing #H = height of tree
            H = cld(cllog2(N + 1), b)
        end

        if b′ === nothing #b′ = cache period
            b′ = b + cllog2(H)
        end

        bkt = undefs(Ti, 1 << b) #bkt = scratch space for cumulative count
        cnt = undefs(Ti, 1 << b, N >> b′ + 1, H) #cnt = cached cumulative counts
        pre = undefs(Tv, 1 << b + 1) #pre = scratch space for cumulative sum
        scn = undefs(Tv, 1 << b + 1, N >> b′ + 1, H) #scn = cached cumulative sums
        wgt = undefs(Tv, N, H) #wgt = permuted copies of weights
        for q = 1:N
            wgt[q, end] = val[q]
        end
        
        byt = undefs(Ti, N)

        for h = H : -1 : 2 #h = level in tree
            for i′ = 1 : 1 << (h * b) : N + 1 #i′ = "quotient" of sorts
                for d = 1:(1 << b)
                    bkt[d] = max(i′ + (d - 1) << ((h - 1) * b) - 1, 1)
                end
                for q = max(i′ - 1, 1) : min(i′ + (1 << (h * b)) - 1, N + 1) - 1
                    i = idx[q]
                    d = i >> ((h - 1) * b) & (1 << b - 1) + 1
                    q′ = bkt[d]
                    byt[q′] = (idx[q′] & ~(1 << ((h - 1) * b) - 1)) | (i & (1 << ((h - 1) * b) - 1))
                    wgt[q′, h - 1] = wgt[q, h]
                    bkt[d] = q′ + 1
                end
            end

            for d = 1 : 1 << b
                cnt[d, 1, h - 1] = 0
            end
            scn[1, 1, h] = 0
            for d = 1 : 1 << b
                scn[d + 1, 1, h] = 0
            end
            zero!(bkt)
            zero!(pre)
            for q = 1 : N
                i = idx[q]
                d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                bkt[d] += 1
                pre[d] += wgt[q, h]
                if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                    Q = (q >> b′) + 1
                    scn[1, Q, h] = 0
                    for d = 1 : 1 << b
                        cnt[d, Q, h - 1] = bkt[d]
                        scn[d + 1, Q, h] = pre[d] + scn[d, Q, h]
                    end
                end
            end
            idx, byt = byt, idx
        end

        scn[1, 1, 1] = 0
        for d = 1 : 1 << b
            scn[d + 1, 1, 1] = 0
        end
        zero!(pre)
        for q = 1 : N
            i = idx[q]
            d = i & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
            pre[d] += wgt[q, 1]
            if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                Q = (q >> b′) + 1
                scn[1, Q, 1] = 0
                for d = 1 : 1 << b
                    scn[d + 1, Q, 1] = pre[d] + scn[d, Q, 1]
                end
            end
        end

        byt = idx
        return SparseSummedRooks{Ti, Tv}(N, b, b′, H, byt, cnt, wgt, scn)
    end
end

function Base.getindex(arg::SparseSummedRooks{Ti, Tv}, i::Integer, j::Integer) where {Ti, Tv}
    @inbounds begin
        N = arg.N
        b = arg.b
        b′ = arg.b′
        H = arg.H
        byt = arg.byt
        cnt = arg.cnt
        wgt = arg.wgt
        scn = arg.scn

        Δq = j - 1
        i = Ti(i - 1)::Ti
        s = Tv(0)
        for h = H : -1 : 2
            i′ = i & ~(1 << (h * b) - 1) + 1
            q₁ = max(i′ - 1, 1) - 1
            q₂ = q₁ + Δq
            d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1
            Q₁ = (q₁ >> b′) + 1
            Q₂ = (q₂ >> b′) + 1
            s += scn[d, Q₂, h] - scn[d, Q₁, h]
            Δq = cnt[d, Q₂, h - 1] - cnt[d, Q₁, h - 1]
            for q = (Q₁ - 1) << b′ + 1 : q₁
                d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                if d′ < d
                    s -= wgt[q, h]
                end
                Δq -= d′ == d
            end
            for q = (Q₂ - 1) << b′ + 1 : q₂
                d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                if d′ < d
                    s += wgt[q, h]
                end
                Δq += d′ == d
            end
        end

        i′ = i & ~(1 << b - 1) + 1
        q₁ = max(i′ - 1, 1) - 1
        q₂ = q₁ + Δq

        d = i & ((1 << b) - 1) + 1
        Q₁ = (q₁ >> b′) + 1
        Q₂ = (q₂ >> b′) + 1
        s += scn[d + 1, Q₂, 1] - scn[d + 1, Q₁, 1]
        for q = (Q₁ - 1) << b′ + 1 : q₁
            d′ = byt[q] & ((1 << b) - 1) + 1
            if d′ <= d
                s -= wgt[q, 1]
            end
        end
        for q = (Q₂ - 1) << b′ + 1 : q₂
            d′ = byt[q] & ((1 << b) - 1) + 1
            if d′ <= d
                s += wgt[q, 1]
            end
        end

        return s
    end
end

SparseCountedRooks(N, idx::Vector{Ti}; kwargs...) where {Ti} = SparseCountedRooks{Ti}(N, idx; kwargs...)

function SparseCountedRooks{Ti}(N, idx::Vector{Ti}; b = nothing, H = nothing, b′ = nothing) where {Ti}
    @inbounds begin
        @debug begin
            @assert length(idx) == N
            hash = falses(N)
            for i in idx
                @assert 1 <= i <= N
                @assert !hash[i]
                hash[i] = true
            end
        end

        if b === nothing #b = branching factor of tree
            if H === nothing
                b = cld(cllog2(N + 1), 3)
            else
                b = cld(cllog2(N + 1), H)
            end
        end

        if H === nothing #H = height of tree
            H = cld(cllog2(N + 1), b)
        end

        if b′ === nothing #b′ = cache period
            b′ = b + cllog2(H)
        end

        bkt = undefs(Ti, 1 << b + 1) #bkt = scratch space for cumulative count
        cnt = undefs(Ti, 1 << b + 1, N >> b′ + 1, H) #cnt = cached cumulative counts
        
        byt = undefs(Ti, N)

        for h = H : -1 : 1 #h = level in tree
            for i′ = 1 : 1 << (h * b) : N + 1 #i′ = "quotient" of sorts
                for d = 1:(1 << b) + 1
                    bkt[d] = max(i′ + (d - 1) << ((h - 1) * b) - 1, 1)
                end
                for q = max(i′ - 1, 1) : min(i′ + (1 << (h * b)) - 1, N + 1) - 1
                    i = idx[q]
                    d = i >> ((h - 1) * b) & (1 << b - 1) + 1
                    q′ = bkt[d]
                    byt[q′] = (idx[q′] & ~(1 << ((h - 1) * b) - 1)) | (i & (1 << ((h - 1) * b) - 1))
                    bkt[d] = q′ + 1
                end
            end
            cnt[1, 1, h] = 0
            for d = 1 : 1 << b
                cnt[d + 1, 1, h] = 0
            end
            zero!(bkt)
            for q = 1 : N
                i = idx[q]
                d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1 #d = a "remainder" of sorts
                bkt[d] += 1
                if q & (1 << b′ - 1) == 0 #Cache bkt at the end of each 1 << b′ block
                    Q = (q >> b′) + 1
                    cnt[1, Q, h] = 0
                    for d = 1 : 1 << b
                        cnt[d + 1, Q, h] = bkt[d] + cnt[d, Q, h]
                    end
                end
            end

            idx, byt = byt, idx
        end
        byt = idx
        return SparseCountedRooks{Ti}(N, b, b′, H, byt, cnt)
    end
end

function Base.getindex(arg::SparseCountedRooks{Ti}, i::Integer, j::Integer) where {Ti}
    @inbounds begin
        N = arg.N
        b = arg.b
        b′ = arg.b′
        H = arg.H
        byt = arg.byt
        cnt = arg.cnt

        Δq = j - 1
        i = Ti(i - 1)::Ti
        s = Ti(0)
        for h = H : -1 : 2
            i′ = i & ~(1 << (h * b) - 1) + 1
            q₁ = max(i′ - 1, 1) - 1
            q₂ = q₁ + Δq
            d = i >> ((h - 1) * b) & ((1 << b) - 1) + 1
            Q₁ = (q₁ >> b′) + 1
            Q₂ = (q₂ >> b′) + 1
            s += cnt[d, Q₂, h] - cnt[d, Q₁, h]
            Δq = (cnt[d + 1, Q₂, h] - cnt[d, Q₂, h]) - (cnt[d + 1, Q₁, h] - cnt[d, Q₁, h])
            msk = ((1 << b) - 1) << ((h - 1) * b)
            cmp = (d - 1) << ((h - 1) * b)
            for q = (Q₁ - 1) << b′ + 1 : q₁
                d′ = byt[q] & msk
                s -= d′ < cmp
                Δq -= d′ == cmp
                #d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                #s -= d′ < d
                #Δq -= d′ == d
            end
            for q = (Q₂ - 1) << b′ + 1 : q₂
                d′ = byt[q] & msk
                s += d′ < cmp
                Δq += d′ == cmp
                #d′ = byt[q] >> ((h - 1) * b) & ((1 << b) - 1) + 1
                #s += d′ < d
                #Δq += d′ == d
            end
        end

        i′ = i & ~(1 << b - 1) + 1
        q₁ = max(i′ - 1, 1) - 1
        q₂ = q₁ + Δq

        d = i & ((1 << b) - 1) + 1
        Q₁ = (q₁ >> b′) + 1
        Q₂ = (q₂ >> b′) + 1
        s += cnt[d + 1, Q₂, 1] - cnt[d + 1, Q₁, 1]
        msk = (1 << b) - 1
        cmp = d - 1
        for q = (Q₁ - 1) << b′ + 1 : q₁
            d′ = byt[q] & msk
            s -= d′ <= cmp
            #d′ = byt[q] & ((1 << b) - 1) + 1
            #s -= d′ <= d
        end
        for q = (Q₂ - 1) << b′ + 1 : q₂
            d′ = byt[q] & msk
            s += d′ <= cmp
            #d′ = byt[q] & ((1 << b) - 1) + 1
            #s += d′ <= d
        end
        return s
    end
end