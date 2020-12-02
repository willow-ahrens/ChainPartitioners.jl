struct MagneticPartitioner end

function partition_stripe(A::SparseMatrixCSC, k, ::MagneticPartitioner, Π; kwargs...)
    @inbounds begin
        (m, n) = size(A)
        Π = convert(MapPartition, Π)
        asg = undefs(Int, n)
        for j = 1:n
            if A.colptr[j] == A.colptr[j + 1]
                asg[j] = rand(1:k)
                #asg[j] = k
            else
                i = A.rowval[rand(A.colptr[j]:A.colptr[j + 1] - 1)]
                #i = A.rowval[A.colptr[j]]
                asg[j] = Π.asg[i]
            end
        end
        return MapPartition(k, asg)
    end
end

struct GreedyBottleneckPartitioner{Mdl}
    mdl::Mdl
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::GreedyBottleneckPartitioner{Mdl}, Π; adj_A = nothing, kwargs...) where {Tv, Ti, Mdl <: AbstractLocalCostModel}
    @inbounds begin
        (m, n) = size(A)
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        x_width = undefs(Ti,K)
        x_work = undefs(Ti,K)
        x_local = undefs(Ti,K)
        x_comm = undefs(Ti,K)
        cst = undefs(eltype(Mdl),K)

        Π_dmn = convert(DomainPartition, Π)
        Π_map = convert(MapPartition, Π)

        m, n = size(A)
        hst = zeros(Ti, n)
        for k = 1:K
            s = Π_dmn.spl[k]
            s′ = Π_dmn.spl[k + 1]
            x_width_k = s′ - s
            x_work_k = 0
            x_local_k = 0
            x_comm_k = 0
            for _s = s:(s′ - 1)
                _i = Π_dmn.prm[_s]
                q = adj_A.colptr[_i]
                q′ = adj_A.colptr[_i + 1]
                x_work_k += q′ - q
                for _q = q : q′ - 1
                    j = adj_A.rowval[_q]
                    if hst[j] < s
                        x_comm_k += 1
                    end
                    hst[j] = s
                end
            end
            x_width[k] = x_width_k
            x_work[k] = x_work_k
            x_local[k] = x_local_k
            x_comm[k] = x_comm_k
            cst[k] = method.mdl(x_width_k, x_work_k, x_local_k, x_comm_k, k)
        end

        asg = ones(Ti, n)

        for j = randperm(n)
            best_c = -Inf
            best_k = 0
            for q = A.colptr[j] : A.colptr[j + 1] - 1
                i = A.rowval[q]
                k = Π_map.asg[i]
                c = cst[k]
                if c > best_c
                    best_c = c
                    best_k = k
                end
            end
            if best_k != 0
                asg[j] = best_k
                x_local[best_k] += 1
                x_comm[best_k] -= 1
                cst[best_k] = method.mdl(x_width[best_k], x_work[best_k], x_local[best_k], x_comm[best_k], best_k)
            end
        end

        return MapPartition(K, asg)
    end
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::GreedyBottleneckPartitioner{Mdl}, Π::SplitPartition; adj_A = nothing, adj_net=nothing, kwargs...) where {Tv, Ti, Mdl <: AbstractLocalCostModel}
    @inbounds begin
        (m, n) = size(A)
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        x_width = undefs(Ti,K)
        x_work = undefs(Ti,K)
        x_local = undefs(Ti,K)
        x_comm = undefs(Ti,K)
        cst = undefs(Float64,K)#TODO we need to do something about these costs

        m, n = size(A)

        if adj_net === nothing
            hst = zeros(Ti, n)
            for k = 1:K
                i = Π.spl[k]
                i′ = Π.spl[k + 1]
                x_width_k = i′ - i
                x_work_k = adj_A.colptr[i′] - adj_A.colptr[i]
                x_local_k = 0
                x_comm_k = 0
                for _i = i:(i′ - 1)
                    q = adj_A.colptr[_i]
                    q′ = adj_A.colptr[_i + 1]
                    for _q = q : q′ - 1
                        j = adj_A.rowval[_q]
                        if hst[j] < i
                            x_comm_k += 1
                        end
                        hst[j] = i
                    end
                end
                x_width[k] = x_width_k
                x_work[k] = x_work_k
                x_local[k] = x_local_k
                x_comm[k] = x_comm_k
                cst[k] = method.mdl(x_width_k, x_work_k, x_local_k, x_comm_k, k)
            end
        else
            for k = 1:K
                i = Π.spl[k]
                i′ = Π.spl[k + 1]
                x_width_k = i′ - i
                x_work_k = adj_A.colptr[i′] - adj_A.colptr[i]
                x_local_k = 0
                x_comm_k = adj_net[i, i′]
                x_width[k] = x_width_k
                x_work[k] = x_work_k
                x_local[k] = x_local_k
                x_comm[k] = x_comm_k
                cst[k] = method.mdl(x_width_k, x_work_k, x_local_k, x_comm_k, k)
            end
        end

        Π = convert(MapPartition, Π)
        asg = ones(Ti, n)

        for j = randperm(n)
            best_c = -Inf
            best_k = 0
            for q = A.colptr[j] : A.colptr[j + 1] - 1
                i = A.rowval[q]
                k = Π.asg[i]
                c = cst[k]
                if c > best_c
                    best_c = c
                    best_k = k
                end
            end
            if best_k != 0
                asg[j] = best_k
                x_local[best_k] += 1
                x_comm[best_k] -= 1
                cst[best_k] = method.mdl(x_width[best_k], x_work[best_k], x_local[best_k], x_comm[best_k], best_k)
            end
        end

        return MapPartition(K, asg)
    end
end