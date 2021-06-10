struct LazyBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

#TODO this file needs to get redistribued, and we should use step oracles for comm costs.

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{F}, args...; kwargs...) where {Tv, Ti, F}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A, args...)

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
                    if Step(f)(Same(j), Next(j′), Same(k)) > c
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

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

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

struct LazyFlipBisectCostBottleneckSplitter{F, T}
    f::F
    ϵ::T
end

#TODO this file needs to get redistribued, and we should use step oracles for comm costs.

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyFlipBisectCostBottleneckSplitter{F}, args...; kwargs...) where {Tv, Ti, F}
    @inbounds begin 
        (m, n) = size(A)
        N = nnz(A)
        ϵ = method.ϵ

        f = oracle_stripe(StepHint(), method.f, A, args...; kwargs...)

        spl = undefs(Int, K + 1)
        spl[1] = 1

        spl_hi = fill(n + 1, K + 1)
        spl_hi[1] = 1

        function probe(c)
            @inbounds begin
                spl[1] = 1
                j = 1
                k = 1

                while f(1, 1, k) <= c
                    if k == K
                        spl[K + 1] = n + 1
                        return true
                    end
                    spl[k + 1] = 1
                    k += 1
                end

                for j′ = 2:n + 1
                    if Step(f)(Same(j), Next(j′), Same(k)) <= c
                        while f(j, j′, k) <= c
                            if k == K
                                spl[K + 1] = n + 1
                                return true
                            end
                            spl[k + 1] = j′
                            j = j′
                            k += 1
                        end
                    end
                end
                return false
            end
        end

        c_lo, c_hi = bound_stripe(A, K, args..., f) ./ 1

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

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractConnectivityModel}, args...; kwargs...) where {Tv, Ti}
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

                n_vertices = 0
                n_pins = 0
                n_net = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += A.colptr[j′ + 1] - A.colptr[j′]
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        if hst[i] < j
                            n_net += 1
                        end
                        cch[q] = hst[i]
                        hst[i] = j′
                    end
                    while k < K && f(n_vertices, n_pins, n_net, k) > c
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = A.colptr[j′ + 1] - A.colptr[j′]
                        n_net = A.colptr[j′ + 1] - A.colptr[j′]
                    end
                end
                res = k < K || f(n_vertices, n_pins, n_net, K) <= c
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

                n_vertices = 0
                n_pins = 0
                n_net = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += A.colptr[j′ + 1] - A.colptr[j′]
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        if cch[q] < j
                            n_net += 1
                        end
                    end
                    while f(n_vertices, n_pins, n_net, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = A.colptr[j′ + 1] - A.colptr[j′]
                        n_net = A.colptr[j′ + 1] - A.colptr[j′]
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

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractMonotonizedSymmetricConnectivityModel}, args...; kwargs...) where {Tv, Ti}
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

                n_vertices = 0
                n_pins = 0
                n_dia_nets = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_pins, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        if hst[i] < j
                            n_dia_nets += 1
                        end
                        cch[q] = hst[i]
                        hst[i] = j′
                    end
                    if hst[j′] < j
                        n_dia_nets += 1
                    end
                    dia[j′] = hst[j′]
                    hst[j′] = j′
                    while k < K && f(n_vertices, n_pins, n_dia_nets, k) > c
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_pins, 0)
                        n_dia_nets = A.colptr[j′ + 1] - A.colptr[j′] + (dia[j′] < j′)
                    end
                end
                res = k < K || f(n_vertices, n_pins, n_dia_nets, K) <= c
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

                n_vertices = 0
                n_pins = 0
                n_dia_nets = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_pins, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        if cch[q] < j
                            n_dia_nets += 1
                        end
                    end
                    if dia[j′] < j
                        n_dia_nets += 1
                    end
                    while f(n_vertices, n_pins, n_dia_nets, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = max(A.colptr[j′ + 1] - A.colptr[j′] - f.Δ_pins, 0)
                        n_dia_nets = A.colptr[j′ + 1] - A.colptr[j′] + (dia[j′] < j′)
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

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::LazyBisectCostBottleneckSplitter{<:AbstractPrimaryConnectivityModel}, Π, args...; kwargs...) where {Tv, Ti}
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

                n_vertices = 0
                n_pins = 0
                n_local_net = 0
                x_comm = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += A.colptr[j′ + 1] - A.colptr[j′]
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
                                n_local_net += 1
                            else
                                x_comm += 1
                            end
                        end
                        hst[i] = j′
                    end
                    while f(n_vertices, n_pins, n_local_net, x_comm, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = A.colptr[j′ + 1] - A.colptr[j′]
                        n_local_net = 0
                        x_comm = A.colptr[j′ + 1] - A.colptr[j′]
                        if drt[k] == j′
                            n_local_net += lcl[k]
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
    Δ_pin = 0
    g(n_vertices, n_pins, n_over_pins, n_net, n_dia_net, n_local_net, k) = begin
        if f isa AbstractNetCostModel
            return f(n_vertices, n_pins, n_net, k)
        elseif f isa AbstractSymCostModel
            Δ_pin = f.Δ_pin
            return f(n_vertices, n_over_pins, n_dia_net, k)
        elseif f isa AbstractPrimaryConnectivityModel
            return f(n_vertices, n_pins, n_local_net, n_net - n_local_net, k)
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

                n_vertices = 0
                n_pins = 0
                n_over_pins = 0
                n_net = 0
                n_dia_net = 0
                n_local_net = 0
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += A.colptr[j′ + 1] - A.colptr[j′]
                    n_over_pins += max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_pin, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        _k = Π_map.asg[i]
                        if drt[_k] < j′
                            lcl[_k] = 0
                        end
                        lcl[_k] += 1
                        drt[_k] = j′
                        if hst[i] < j
                            n_net += 1
                            if _k == k
                                n_local_net += 1
                            end
                        end
                        if (i < j || i >= j′) && hst[i] < j
                            n_dia_net += 1
                        end
                        hst[i] = j′
                    end
                    if j′ <= m && hst[j′] < j
                        n_dia_net += 1
                    end
                    while g(n_vertices, n_pins, n_over_pins, n_net, n_dia_net, n_local_net, k) > c
                        if k == K
                            return false
                        end
                        spl[k + 1] = j′
                        j = j′
                        k += 1
                        n_vertices = 1
                        n_pins = A.colptr[j′ + 1] - A.colptr[j′]
                        n_net = A.colptr[j′ + 1] - A.colptr[j′]
                        n_dia_net = A.colptr[j′ + 1] - A.colptr[j′] + (j′ <= m && hst[j′] < j′)
                        n_over_pins = max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_pin, 0)
                        n_local_net = 0
                        if drt[k] == j′
                            n_local_net += lcl[k]
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
    Δ_pin = 0
    g(n_vertices, n_pins, n_over_pins, n_net, n_dia_net, n_local_net, k) = begin
        if f isa AbstractNetCostModel
            return f(n_vertices, n_pins, n_net, k)
        elseif f isa AbstractSymCostModel
            Δ_pin = f.Δ_pin
            return f(n_vertices, n_over_pins, n_dia_net, k)
        elseif f isa AbstractPrimaryConnectivityModel
            return f(n_vertices, n_pins, n_local_net, n_net - n_local_net, k)
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

                n_vertices = 0
                n_pins = 0
                n_over_pins = 0
                n_net = 0
                n_dia_net = 0
                n_local_net = 0
                while g(n_vertices, n_pins, n_over_pins, n_net, n_dia_net, n_local_net, k) <= c
                    if k == K
                        return true
                    end
                    spl[k + 1] = j
                    k += 1
                end
                for j′ = 1:n
                    n_vertices += 1
                    n_pins += A.colptr[j′ + 1] - A.colptr[j′]
                    n_over_pins += max(A.colptr[j′ + 1] - A.colptr[j′] - Δ_pin, 0)
                    for q = A.colptr[j′]:A.colptr[j′ + 1] - 1
                        i = A.rowval[q]
                        _k = Π_map.asg[i]
                        if drt[_k] < j′
                            lcl[_k] = 0
                        end
                        lcl[_k] += 1
                        drt[_k] = j′
                        if hst[i] < j
                            n_net += 1
                            if _k == k
                                n_local_net += 1
                            end
                        end
                        if (i < j || i >= j′) && hst[i] < j
                            n_dia_net += 1
                        end
                        hst[i] = j′
                    end
                    if j′ <= m && hst[j′] < j
                        n_dia_net += 1
                    end
                    while g(n_vertices, n_pins, n_over_pins, n_net, n_dia_net, n_local_net, k) <= c
                        if k == K
                            return true
                        end
                        spl[k + 1] = j′ + 1
                        j = j′ + 1
                        k += 1
                        n_vertices = 0
                        n_pins = 0
                        n_net = 0
                        n_dia_net = 0
                        n_over_pins = 0
                        n_local_net = 0
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