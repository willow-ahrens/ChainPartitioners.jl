struct DominanceSum{Tv, Ti} <: AbstractMatrix{Tv}
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

mutable struct SparseStepwiseDominanceSum{Tv, Ti} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    N::Int
    i::Ti
    j::Ti
    pos::Vector{Ti}
    idx::Vector{Ti}
    val::Vector{Tv}
    Δ::Vector{Ti}
    c::Tv
end

Base.size(arg::DominanceSum) = (arg.m + 1, arg.n + 1)

dominancesum(args...; kwargs...) = dominancesum(NoHint(), args...; kwargs...)
dominancesum(::AbstractHint, args...; kwargs...) = @assert false
function dominancesum(hint::AbstractHint, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    (m, n) = size(A)
    N = nnz(A)
    pos = copy(A.colptr)
    idx = copy(A.rowval)
    val = A.nzval
    return dominancesum!(hint, m, n, N, pos, idx, val; kwargs...)
end

dominancesum!(args...; kwargs...) = dominancesum!(NoHint(), args...; kwargs...)
dominancesum!(::AbstractHint, args...; kwargs...) = @assert false
function dominancesum!(hint::AbstractHint, m, n, N, pos, idx, val; H = nothing, b = nothing, b′ = nothing, kwargs...)
    if H === b === b′ === nothing
        return DominanceSum(hint, m, n, N, pos, idx, val; b = 4, kwargs...)
    else
        return DominanceSum(hint, m, n, N, pos, idx, val; H = H, b = b, b′ = b′, kwargs...)
    end
end
function dominancesum!(hint::SparseHint, m, n, N, pos, idx, val; kwargs...)
    return DominanceSum(hint, m, n, N, pos, idx, val; kwargs...)
end
function dominancesum!(hint::StepHint, m, n, N, pos, idx, val; kwargs...)
    SparseStepwiseDominanceCount(hint, m, n, N, pos, idx, val; kwargs...)
end

DominanceSum(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Tv, Ti} =
    DominanceSum{Tv, Ti}(hint, m, n, N, pos, idx, val; kwargs...)
function DominanceSum{Tv, Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; b = nothing, H = nothing, b′ = nothing) where {Tv, Ti}
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

        return DominanceSum{Tv, Ti}(m, n, N, b, b′, H, pos, qos, byt, cnt, wgt, scn)
    end
end

Base.getindex(arg::DominanceSum{Tv, Ti}, i::Integer, j::Integer) where {Tv, Ti} = arg(i, j)
function (arg::DominanceSum{Tv, Ti})(i::Integer, j::Integer) where {Tv, Ti}
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



SparseStepwiseDominanceSum(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Tv, Ti} =
    SparseStepwiseDominanceSum{Tv, Ti}(hint, m, n, N, pos, idx, val; kwargs...)
function SparseStepwiseDominanceSum{Tv, Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Tv, Ti}
    @inbounds begin
        i = j = 0
        Δ = zeros(Ti, m)
        c = Tv(0)
        return SparseStepwiseDominanceSum(m, n, N, i, j, pos, idx, val, Δ, c)
    end
end

Base.getindex(arg::SparseStepwiseDominanceSum{Tv, Ti}, i::Integer, j::Integer) where {Tv, Ti} = arg(i, j)
function (arg::SparseStepwiseDominanceSum{Tv, Ti})(i::Integer, j::Integer) where {Tv, Ti}
    @inbounds begin
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        val = arg.val
        arg_i = arg.i
        arg_j = arg.j
        #reset case
        if (arg.m + pos[j + 1]) < pos[arg_j + 1] - pos[j + 1]
            arg_j = zero(Ti)
            c = zero(Tv)
            zero!(Δ)
        end
        for q = pos[j + 1] : pos[arg_j + 1] - 1
            Δ[idx[q]] -= val[q]
            if idx[q] <= i
                c -= val[q]
            end
        end
        for q = pos[arg_j + 1] : pos[j + 1] - 1
            Δ[idx[q]] += val[q]
            if idx[q] <= i
                c += val[q]
            end
        end
        for q = i + 1 : arg_i
            c -= Δ[q]
        end
        for q = arg_i + 1 : i
            c += Δ[q]
        end
        arg.i = i
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(i::Same, j::Same) where {Tv, Ti, Cnt <: SparseStepwiseDominanceSum{Tv, Ti}}
    begin
        arg = stp.ocl
        return arg.c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Same, _j::Next) where {Tv, Ti, Cnt <: SparseStepwiseDominanceSum{Tv, Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        for q = pos[j] : pos[j + 1] - 1
            Δ[idx[q]] += val[q]
            if idx[q] <= i
                c += val[q]
            end
        end
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Same, _j::Prev) where {Tv, Ti, Cnt <: SparseStepwiseDominanceSum{Tv, Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        val = arg.val
        for q = pos[j + 1] : pos[j + 2] - 1
            Δ[idx[q]] -= val[q]
            if idx[q] <= i
                c -= val[q]
            end
        end
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Next, _j::Same) where {Tv, Ti, Cnt <: SparseStepwiseDominanceSum{Tv, Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        c += Δ[i]
        arg.i = i
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Prev, _j::Same) where {Tv, Ti, Cnt <: SparseStepwiseDominanceSum{Tv, Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        c -= Δ[i + 1]
        arg.i = i
        arg.c = c
        return c
    end
end



struct DominanceCount{Ti} <: AbstractMatrix{Ti}
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

Base.size(arg::DominanceCount) = (arg.m + 1, arg.n + 1)

struct BinaryDominanceCount{Ti, Tb} <: AbstractMatrix{Ti}
    m::Int
    n::Int
    N::Int
    H::Int
    pos::Vector{Ti}
    qos::Vector{Ti}
    byt::Array{Tb, 2}
    cnt::Array{Ti, 2}
end

Base.size(arg::BinaryDominanceCount) = (arg.m + 1, arg.n + 1)

mutable struct SparseStepwiseDominanceCount{Ti} <: AbstractMatrix{Ti}
    m::Int
    n::Int
    N::Int
    i::Ti
    j::Ti
    pos::Vector{Ti}
    idx::Vector{Ti}
    Δ::Vector{Ti}
    c::Ti
end

Base.size(arg::SparseStepwiseDominanceCount) = (arg.m + 1, arg.n + 1)

dominancecount(args...; kwargs...) = dominancecount(NoHint(), args...; kwargs...)
dominancecount(::AbstractHint, args...; kwargs...) = @assert false
function dominancecount(hint::AbstractHint, A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
    @inbounds begin
        (m, n) = size(A)

        return dominancecount!(hint, m, n, nnz(A), copy(A.colptr), copy(A.rowval); kwargs...)
    end
end

dominancecount!(args...; kwargs...) = dominancecount!(NoHint(), args...; kwargs...)
dominancecount!(::AbstractHint, args...; kwargs...) = @assert false
function dominancecount!(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    BinaryDominanceCount(hint, m, n, N, pos, idx; kwargs...)
end
function dominancecount!(hint::SparseHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    DominanceCount(hint, m, n, N, pos, idx; kwargs...)
end
function dominancecount!(hint::StepHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    SparseStepwiseDominanceCount(hint, m, n, N, pos, idx; kwargs...)
end

DominanceCount(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} = 
    DominanceCount{Ti}(hint, m, n, N, pos, idx; kwargs...)
function DominanceCount{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; b = nothing, H = nothing, b′ = nothing, kwargs...) where {Ti}
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
        return DominanceCount{Ti}(m, n, N, b, b′, H, pos, qos, byt, cnt)
    end
end

Base.getindex(arg::DominanceCount{Ti}, i::Integer, j::Integer) where {Ti} = arg(i, j)
function (arg::DominanceCount{Ti})(i::Integer, j::Integer) where {Ti}
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

BinaryDominanceCount(hint::AbstractHint, m, n, N, pos, idx; kwargs...) =
    BinaryDominanceCount{UInt}(hint, m, n, N, pos, idx; kwargs...)
BinaryDominanceCount{Tb}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Tb, Ti} =
    BinaryDominanceCount{Tb, Ti}(hint, m, n, N, pos, idx; kwargs...)
function BinaryDominanceCount{Tb, Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Tb, Ti}
    @inbounds begin
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
            _cnt = 0
            for i′ = 1 : 1 << h : m + 1 #i′ = "quotient" of sorts
                bkt_1 = 0
                bkt_2 = 0
                for q = qos[i′] : qos[min(i′ + (1 << h), end)] - 1 #q = position in current level
                    i = idx[q]
                    d = i >> (h - 1) & 1 #d = a "remainder" of sorts
                    Q = ((q - 1) >> log2nbits(Tb)) + 1
                    byt[Q, h] |= d << ((q - 1) & (nbits(Tb) - 1))
                    cnt[Q + 1, h] = _cnt += d
                    bkt_2 += 1 - d
                end
                bkt_1 = qos[i′]
                bkt_2 += bkt_1
                for q = qos[i′] : qos[min(i′ + (1 << h), end)] - 1
                    i = idx[q]
                    d = i >> (h - 1) & 1
                    q′ = ifelse(d == 0, bkt_1, bkt_2)
                    idx′[q′] = i
                    bkt_1 += 1 - d
                    bkt_2 += d
                end
                qos[min(i′ + 1 << (h - 1), end)] = bkt_1
            end

            idx, idx′ = idx′, idx
        end
        return BinaryDominanceCount{Ti, Tb}(m, n, N, H, pos, qos, byt, cnt)
    end
end

#we have removed b′ because increasing it doesn't significantly reduce the storage cost or preprocessing cost of the structure.

Base.getindex(arg::BinaryDominanceCount{Ti, Tb}, i::Integer, j::Integer) where {Ti, Tb} = arg(i, j)
function (arg::BinaryDominanceCount{Ti, Tb})(i::Integer, j::Integer) where {Ti, Tb}
    @inbounds begin
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
            bkt_2 += count_ones(byt[Q₂, h] & ((one(Tb) << (q₂ & (nbits(Tb) - 1))) - 1))
            bkt_2 -= count_ones(byt[Q₁, h] & ((one(Tb) << (q₁ & (nbits(Tb) - 1))) - 1))
            bkt_1 = Δq - bkt_2
            s += ifelse(d == 0, 0, bkt_1)
            Δq = ifelse(d == 0, bkt_1, bkt_2)
        end

        return s + Δq
    end
end



SparseStepwiseDominanceCount(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti} =
    SparseStepwiseDominanceCount{Ti}(hint, m, n, N, pos, idx; kwargs...)
function SparseStepwiseDominanceCount{Ti}(hint::AbstractHint, m, n, N, pos::Vector{Ti}, idx::Vector{Ti}; kwargs...) where {Ti}
    @inbounds begin
        i = j = 0
        Δ = zeros(Ti, m)
        c = Ti(0)
        return SparseStepwiseDominanceCount(m, n, N, i, j, pos, idx, Δ, c)
    end
end

Base.getindex(arg::SparseStepwiseDominanceCount{Ti}, i::Integer, j::Integer) where {Ti} = arg(i, j)
function (arg::SparseStepwiseDominanceCount{Ti})(i::Integer, j::Integer) where {Ti}
    @inbounds begin
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        arg_i = arg.i
        arg_j = arg.j
        #reset case
        if (arg.m + pos[j + 1]) < pos[arg_j + 1] - pos[j + 1]
            arg_j = zero(Ti)
            c = zero(Ti)
            zero!(Δ)
        end
        for q = pos[j + 1] : pos[arg_j + 1] - 1
            Δ[idx[q]] -= 1
            c -= idx[q] <= arg_i
        end
        for q = pos[arg_j + 1] : pos[j + 1] - 1
            Δ[idx[q]] += 1
            c += idx[q] <= arg_i
        end
        for q = i + 1 : arg_i
            c -= Δ[q]
        end
        for q = arg_i + 1 : i
            c += Δ[q]
        end
        arg.i = i
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(i::Same, j::Same) where {Ti, Cnt <: SparseStepwiseDominanceCount{Ti}}
    begin
        arg = stp.ocl
        return arg.c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Same, _j::Next) where {Ti, Cnt <: SparseStepwiseDominanceCount{Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        for q = pos[j] : pos[j + 1] - 1
            Δ[idx[q]] += 1
            c += idx[q] <= i
        end
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Same, _j::Prev) where {Ti, Cnt <: SparseStepwiseDominanceCount{Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        pos = arg.pos
        idx = arg.idx
        for q = pos[j + 1] : pos[j + 2] - 1
            Δ[idx[q]] -= 1
            c -= idx[q] <= i
        end
        arg.j = j
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Next, _j::Same) where {Ti, Cnt <: SparseStepwiseDominanceCount{Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        c += Δ[i]
        arg.i = i
        arg.c = c
        return c
    end
end

@propagate_inbounds function (stp::Step{Cnt})(_i::Prev, _j::Same) where {Ti, Cnt <: SparseStepwiseDominanceCount{Ti}}
    begin
        arg = stp.ocl
        i = destep(_i)
        j = destep(_j)
        i -= 1
        j -= 1
        c = arg.c
        Δ = arg.Δ
        c -= Δ[i + 1]
        arg.i = i
        arg.c = c
        return c
    end
end



struct RookSum{Ti, Tv} <: AbstractMatrix{Tv}
    N::Int
    b::Int
    b′::Int
    H::Int
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
    wgt::Array{Tv, 2}
    scn::Array{Tv, 3}
end

Base.size(arg::RookSum) = (arg.N + 1, arg.N + 1)

rooksum!(args...; kwargs...) = rooksum!(NoHint(), args...; kwargs...)
rooksum!(::AbstractHint, args...; kwargs...) = @assert false
function rooksum!(hint::AbstractHint, N, idx, val; H = nothing, b = nothing, b′ = nothing, kwargs...)
    if H === b === b′ === nothing
        return RookSum(hint, N, idx, val; b = 4, kwargs...)
    else
        return RookSum(hint, N, idx, val; H = H, b = b, b′ = b′, kwargs...)
    end
end
function rooksum!(hint::SparseHint, N, idx, val; kwargs...)
    return RookSum(hint, N, idx, val; kwargs...)
end

RookSum(hint::AbstractHint, N, idx::Vector{Ti}, val::Vector{Tv}; kwargs...) where {Ti, Tv} =
    RookSum{Ti, Tv}(hint, N, idx, val; kwargs...)
function RookSum{Ti, Tv}(hint::AbstractHint, N, idx::Vector{Ti}, val::Vector{Tv}; b = nothing, H = nothing, b′ = nothing, kwargs...) where {Ti, Tv}
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
        return RookSum{Ti, Tv}(N, b, b′, H, byt, cnt, wgt, scn)
    end
end

Base.getindex(arg::RookSum{Ti, Tv}, i::Integer, j::Integer) where {Ti, Tv} = arg(i, j)
function (arg::RookSum{Ti, Tv})(i::Integer, j::Integer) where {Ti, Tv}
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



struct RookCount{Ti} <: AbstractMatrix{Ti}
    N::Int
    b::Int
    b′::Int
    H::Int
    byt::Vector{Ti}
    cnt::Array{Ti, 3}
end

Base.size(arg::RookCount) = (arg.N + 1, arg.N + 1)

struct BinaryRookCount{Tb, Ti} <: AbstractMatrix{Ti}
    N::Int
    H::Int
    byt::Array{Tb, 2}
    cnt::Array{Ti, 2}
end

Base.size(arg::BinaryRookCount) = (arg.N + 1, arg.N + 1)

mutable struct SparseRookStepCounter{Ti} <: AbstractMatrix{Ti}
    N::Int
    i::Ti
    j::Ti
    idx::Vector{Ti}
    Δ::Vector{Ti}
    c::Ti
end

Base.size(arg::SparseRookStepCounter) = (arg.N + 1, arg.N + 1)

rookcount!(args...; kwargs...) = rookcount!(NoHint(), args...; kwargs...)
rookcount!(::AbstractHint, args...; kwargs...) = @assert false
function rookcount!(hint::AbstractHint, N, idx; kwargs...)
    return BinaryRookCount(hint, N, idx; kwargs...)
end
function rookcount!(hint::SparseHint, N, idx; kwargs...)
    RookCount(hint, N, idx; kwargs...)
end

RookCount(hint::AbstractHint, N, idx::Vector{Ti}; kwargs...) where {Ti} =
    RookCount{Ti}(hint, N, idx; kwargs...)
function RookCount{Ti}(hint::AbstractHint, N, idx::Vector{Ti}; b = nothing, H = nothing, b′ = nothing, kwargs...) where {Ti}
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
        return RookCount{Ti}(N, b, b′, H, byt, cnt)
    end
end

Base.getindex(arg::RookCount{Ti}, i::Integer, j::Integer) where {Ti} = arg(i, j)
function (arg::RookCount{Ti})(i::Integer, j::Integer) where {Ti}
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

BinaryRookCount(hint::AbstractHint, N, idx; kwargs...) =
    BinaryRookCount{UInt}(hint, N, idx; kwargs...)
BinaryRookCount{Tb}(hint::AbstractHint, N, idx::Vector{Ti}; kwargs...) where {Tb, Ti} =
    BinaryRookCount{Tb, Ti}(hint, N, idx; kwargs...)
function BinaryRookCount{Tb, Ti}(hint::AbstractHint, N, idx::Vector{Ti}; kwargs...) where {Tb, Ti}
    @inbounds begin
        #b = branching factor of tree = 1

        #H = height of tree
        H = cllog2(N + 1)

        cnt = zeros(Ti, 1 + cld(N, nbits(Tb)), H) #cnt = cached cumulative counts
        
        byt = zeros(Tb, 1 + cld(N, nbits(Tb)), H)
        idx′ = undefs(Ti, N)

        for h = H : -1 : 1 #h = level in tree
            _cnt = 0
            for i′ = 1 : 1 << h : N + 1 #i′ = "quotient" of sorts
                bkt_1 = max(i′ - 1, 1)
                bkt_2 = max(i′ + 1 << (h - 1) - 1, 1)
                for q = max(i′ - 1, 1) : min(i′ + (1 << h) - 1, N + 1) - 1
                    i = idx[q]
                    d = i >> (h - 1) & 1
                    q′ = ifelse(d == 0, bkt_1, bkt_2)
                    idx′[q′] = i
                    bkt_1 += 1 - d
                    bkt_2 += d

                    Q = ((q - 1) >> log2nbits(Tb)) + 1
                    byt[Q, h] |= d << ((q - 1) & (nbits(Tb) - 1))
                    cnt[Q + 1, h] = _cnt += d
                end
            end

            idx, idx′ = idx′, idx
        end
        return BinaryRookCount{Tb, Ti}(N, H, byt, cnt)
    end
end

Base.getindex(arg::BinaryRookCount{Tb, Ti}, i::Integer, j::Integer) where {Ti, Tb} = arg(i, j)
function (arg::BinaryRookCount{Tb, Ti})(i::Integer, j::Integer) where {Ti, Tb}
    @inbounds begin
        H = arg.H
        byt = arg.byt
        cnt = arg.cnt

        Δq = j - 1
        i = Ti(i - 1)::Ti
        s = Ti(0)
        for h = H : -1 : 1
            i′ = i & ~(1 << h - 1) + 1
            q₁ = max(i′ - 1, 1) - 1
            q₂ = q₁ + Δq
            d = i >> (h - 1) & Ti(1)

            Q₁ = (q₁ >> log2nbits(Tb)) + 1
            Q₂ = (q₂ >> log2nbits(Tb)) + 1
            bkt_2 = cnt[Q₂, h] - cnt[Q₁, h]
            bkt_2 += count_ones(byt[Q₂, h] & ((one(Tb) << (q₂ & (nbits(Tb) - 1))) - 1))
            bkt_2 -= count_ones(byt[Q₁, h] & ((one(Tb) << (q₁ & (nbits(Tb) - 1))) - 1))
            bkt_1 = Δq - bkt_2
            s += ifelse(d == 0, 0, bkt_1)
            Δq = ifelse(d == 0, bkt_1, bkt_2)
        end

        return s + Δq
    end
end