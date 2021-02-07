struct LazyBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

#TODO this file needs to get redistribued, and we should use step oracles for comm costs.

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractNetCostModel}, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                f(1, 1, 1)

                for j′ = 2:n + 1
                    if Step(f, Same(), Next(), Same())(j, j′, k) > c
                        while true
                            if k == K
                                return false
                            end
                            spl[k + 1] = j′ - 1
                            j = j′ - 1
                            k += 1
                            if f(j, j′, k) <= c
                                break
                            end
                        end
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., method.f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(1, 1, k))
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

#=
function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractNetCostModel}, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                f(1, 1, 1)

                for j′ = 2:n + 1
                    if Step(f, Same(), Next(), Same())(j, j′, k) > c
                        while true
                            if k == K
                                return false
                            end
                            spl[k + 1] = j′ - 1
                            j = j′ - 1
                            k += 1
                            if f(j, j′, k) <= c
                                break
                            end
                        end
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., method.f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(1, 1, k))
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

#=
function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractNetCostModel}, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ
        f = method.f

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        hst = zeros(Int, m)
        cch = undefs(Ti, N)

        function probe_init(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_net = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += A.colptr[j′ + 1] - A.colptr[j′]
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        if hst[i] < j
                            x_net += 1
                        end
                        cch[q] = hst[i]
                        hst[i] = j′
                    end
                    while k < K && f(x_width, x_work, x_net, k) > c
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = A.colptr[j′ + 1] - A.colptr[j′]
                        x_net = A.colptr[j′ + 1] - A.colptr[j′]
                    end
                end
                res = k < K || f(x_width, x_work, x_net, K) <= c
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return res
            end
        end

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_net = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += A.colptr[j′ + 1] - A.colptr[j′]
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        if cch[q] < j
                            x_net += 1
                        end
                    end
                    while f(x_width, x_work, x_net, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = A.colptr[j′ + 1] - A.colptr[j′]
                        x_net = A.colptr[j′ + 1] - A.colptr[j′]
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(0, 0, 0, k))
        end

        if c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe_init(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end
=#

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractSymCostModel}, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        @assert m == n
        N = nnz(A)
        ϵ = method.ϵ
        f = method.f

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        hst = zeros(Int, m)
        dia = zeros(Int, n)
        cch = undefs(Ti, N)

        function probe_init(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_net = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_work, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        if hst[i] < j
                            x_net += 1
                        end
                        cch[q] = hst[i]
                        hst[i] = j′
                    end
                    if hst[j′] < j
                        x_net += 1
                    end
                    dia[j′] = hst[j′]
                    hst[j′] = j′
                    while k < K && f(x_width, x_work, x_net, k) > c
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_work, 0)
                        x_net = A.colptr[j′ + 1] - A.colptr[j′] + (dia[j′] < j′)
                    end
                end
                res = k < K || f(x_width, x_work, x_net, K) <= c
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return res
            end
        end

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_net = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_work, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        if cch[q] < j
                            x_net += 1
                        end
                    end
                    if dia[j′] < j
                        x_net += 1
                    end
                    while f(x_width, x_work, x_net, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_work, 0)
                        x_net = A.colptr[j′ + 1] - A.colptr[j′] + (dia[j′] < j′)
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(0, 0, 0, k))
        end

        if c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe_init(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            if probe(c)
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractCommCostModel}, Π, args...; kwargs...) where {Tv, Ti}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ
        f = method.f
        Π_map = convert(MapPartition, Π)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        hst = undefs(Int, m)
        lcl = undefs(Int, K)
        drt = undefs(Int, K)

        function probe(c)
            @inbounds begin
                zero!(hst)
                zero!(drt)
                spl[1] = 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_local = 0
                x_comm = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += A.colptr[j′ + 1] - A.colptr[j′]
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        _k = Π_map.asg[i]
                        if drt[_k] < j′
                            lcl[_k] = 0
                        end
                        lcl[_k] += 1
                        drt[_k] = j′
                        if hst[i] < j
                            if _k == k
                                x_local += 1
                            else
                                x_comm += 1
                            end
                        end
                        hst[i] = j′
                    end
                    while f(x_width, x_work, x_local, x_comm, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = A.colptr[j′ + 1] - A.colptr[j′]
                        x_local = 0
                        x_comm = A.colptr[j′ + 1] - A.colptr[j′]
                        if drt[k] == j′
                            x_local += lcl[k]
                            x_comm -= lcl[k]
                        end
                    end
                end
                while k <= K
                    spl[k + 1] = n + 1
                    k += 1
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, Π, args..., f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, f(0, 0, 0, 0, k))
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            chk = probe(c)
            if chk && spl[end] == n + 1
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

#=
function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter, args...; kwargs...) where {Tv, Ti}
    g = nothing
    f = method.f
    Δ_work = 0
    g(x_width, x_work, x_overwork, x_net, x_diagonal, x_local, k) = begin
        if f isa AbstractNetCostModel
            return f(x_width, x_work, x_net, k)
        elseif f isa AbstractSymCostModel
            Δ_work = f.Δ_work
            return f(x_width, x_overwork, x_diagonal, k)
        elseif f isa AbstractCommCostModel
            return f(x_width, x_work, x_local, x_net - x_local, k)
        else
            @assert false
        end
    end
    Π = partition_stripe(SparseMatrixCSC(A'), K, EquiSplitter())
    if length(args) > 0
        Π = args[1]
    end
    
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ
        f = method.f
        Π_map = convert(MapPartition, Π)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        hst = undefs(Int, m)
        lcl = undefs(Int, K)
        drt = undefs(Int, K)

        function probe(c)
            @inbounds begin
                zero!(hst)
                zero!(drt)
                spl[1] = 1
                spl[2:K + 1] .= n + 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_overwork = 0
                x_net = 0
                x_diagonal = 0
                x_local = 0
                for j′ = 1:n
                    x_width += 1
                    x_work += A.colptr[j′ + 1] - A.colptr[j′]
                    x_overwork += max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_work, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        _k = Π_map.asg[i]
                        if drt[_k] < j′
                            lcl[_k] = 0
                        end
                        lcl[_k] += 1
                        drt[_k] = j′
                        if hst[i] < j
                            x_net += 1
                            if _k == k
                                x_local += 1
                            end
                        end
                        if (i < j || i >= j′) && hst[i] < j
                            x_diagonal += 1
                        end
                        hst[i] = j′
                    end
                    if j′ <= m && hst[j′] < j
                        x_diagonal += 1
                    end
                    while g(x_width, x_work, x_overwork, x_net, x_diagonal, x_local, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        x_width = 1
                        x_work = A.colptr[j′ + 1] - A.colptr[j′]
                        x_net = A.colptr[j′ + 1] - A.colptr[j′]
                        x_diagonal = A.colptr[j′ + 1] - A.colptr[j′] + (j′ <= m && hst[j′] < j′)
                        x_overwork = max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_work, 0)
                        x_local = 0
                        if drt[k] == j′
                            x_local += lcl[k]
                        end
                    end
                end
                return true
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        for k = 1:K
            c_lo = max(c_lo, g(0, 0, 0, 0, 0, 0, k))
        end

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            chk = probe(c)
            if chk && spl[end] == n + 1
                c_hi = c
                spl_hi .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_hi)
    end
end

struct LazyFlipBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyFlipBisectCostBottleneckSplitter, args...; kwargs...) where {Tv, Ti}
    g = nothing
    f = method.f
    Δ_work = 0
    g(x_width, x_work, x_overwork, x_net, x_diagonal, x_local, k) = begin
        if f isa AbstractNetCostModel
            return f(x_width, x_work, x_net, k)
        elseif f isa AbstractSymCostModel
            Δ_work = f.Δ_work
            return f(x_width, x_overwork, x_diagonal, k)
        elseif f isa AbstractCommCostModel
            return f(x_width, x_work, x_local, x_net - x_local, k)
        else
            @assert false
        end
    end
    Π = partition_stripe(SparseMatrixCSC(A'), K, EquiSplitter())
    if length(args) > 0
        Π = args[1]
    end
    
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ
        f = method.f
        Π_map = convert(MapPartition, Π)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_lo = fill(n + 1, K + 1)
        spl_lo[1] = 1

        hst = undefs(Int, m)
        lcl = undefs(Int, K)
        drt = undefs(Int, K)

        function probe(c)
            @inbounds begin
                zero!(hst)
                zero!(drt)
                spl[1] = 1
                spl[2:K + 1] .= n + 1
                j = 1
                k = 1

                x_width = 0
                x_work = 0
                x_overwork = 0
                x_net = 0
                x_diagonal = 0
                x_local = 0
                while g(x_width, x_work, x_overwork, x_net, x_diagonal, x_local, k) <= c
                    if k == K
                        return true
                    end
                    spl[k + 1] = j
                    k += 1
                end
                for j′ = 1:n
                    x_width += 1
                    x_work += A.colptr[j′ + 1] - A.colptr[j′]
                    x_overwork += max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_work, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        _k = Π_map.asg[i]
                        if drt[_k] < j′
                            lcl[_k] = 0
                        end
                        lcl[_k] += 1
                        drt[_k] = j′
                        if hst[i] < j
                            x_net += 1
                            if _k == k
                                x_local += 1
                            end
                        end
                        if (i < j || i >= j′) && hst[i] < j
                            x_diagonal += 1
                        end
                        hst[i] = j′
                    end
                    if j′ <= m && hst[j′] < j
                        x_diagonal += 1
                    end
                    while g(x_width, x_work, x_overwork, x_net, x_diagonal, x_local, k) <= c
                        if k == K
                            return true
                        end
                        spl[k + 1] = j′ + 1
                        j = j′ + 1
                        k += 1
                        x_width = 0
                        x_work = 0
                        x_net = 0
                        x_diagonal = 0
                        x_overwork = 0
                        x_local = 0
                    end
                end
                return false
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

        while c_lo * (1 + ϵ) < c_hi
            c = (c_lo + c_hi) / 2
            chk = probe(c)
            if chk && spl[end] == n + 1
                c_hi = c
                spl_lo .= spl
            else
                c_lo = c
            end
        end
        return SplitPartition(K, spl_lo)
    end
end
=#