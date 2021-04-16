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

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::GreedyBottleneckPartitioner{Mdl}, Π; adj_A = nothing, kwargs...) where {Tv, Ti, Mdl <: AbstractSecondaryConnectivityModel}
    @inbounds begin
        (m, n) = size(A)
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        n_vertices = undefs(Ti,K)
        n_pins = undefs(Ti,K)
        n_local_nets = undefs(Ti,K)
        n_remote_nets = undefs(Ti,K)
        cst = undefs(cost_type(method.mdl),K)

        Π_dmn = convert(DomainPartition, Π)
        Π_map = convert(MapPartition, Π)

        m, n = size(A)
        hst = zeros(Ti, n)
        for k = 1:K
            s = Π_dmn.spl[k]
            s′ = Π_dmn.spl[k + 1]
            n_vertices_k = s′ - s
            n_pins_k = 0
            n_local_nets_k = 0
            n_remote_nets_k = 0
            for _s = s:(s′ - 1)
                _i = Π_dmn.prm[_s]
                q = adj_A.colptr[_i]
                q′ = adj_A.colptr[_i + 1]
                n_pins_k += q′ - q
                for _q = q : q′ - 1
                    j = adj_A.rowval[_q]
                    if hst[j] < s
                        n_remote_nets_k += 1
                    end
                    hst[j] = s
                end
            end
            n_vertices[k] = n_vertices_k
            n_pins[k] = n_pins_k
            n_local_nets[k] = n_local_nets_k
            n_remote_nets[k] = n_remote_nets_k
            cst[k] = method.mdl(n_vertices_k, n_pins_k, n_local_nets_k, n_remote_nets_k, k)
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
                n_local_nets[best_k] += 1
                n_remote_nets[best_k] -= 1
                cst[best_k] = method.mdl(n_vertices[best_k], n_pins[best_k], n_local_nets[best_k], n_remote_nets[best_k], best_k)
            end
        end

        return MapPartition(K, asg)
    end
end

function partition_stripe(A::SparseMatrixCSC{Tv, Ti}, K, method::GreedyBottleneckPartitioner{Mdl}, Π::SplitPartition; adj_A = nothing, adj_net=nothing, kwargs...) where {Tv, Ti, Mdl <: AbstractSecondaryConnectivityModel}
    @inbounds begin
        (m, n) = size(A)
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        n_vertices = undefs(Ti,K)
        n_pins = undefs(Ti,K)
        n_local_nets = undefs(Ti,K)
        n_remote_nets = undefs(Ti,K)
        cst = undefs(cost_type(method.mdl),K)

        m, n = size(A)

        if adj_net === nothing
            hst = zeros(Ti, n)
            for k = 1:K
                i = Π.spl[k]
                i′ = Π.spl[k + 1]
                n_vertices_k = i′ - i
                n_pins_k = adj_A.colptr[i′] - adj_A.colptr[i]
                n_local_nets_k = 0
                n_remote_nets_k = 0
                for _i = i:(i′ - 1)
                    q = adj_A.colptr[_i]
                    q′ = adj_A.colptr[_i + 1]
                    for _q = q : q′ - 1
                        j = adj_A.rowval[_q]
                        if hst[j] < i
                            n_remote_nets_k += 1
                        end
                        hst[j] = i
                    end
                end
                n_vertices[k] = n_vertices_k
                n_pins[k] = n_pins_k
                n_local_nets[k] = n_local_nets_k
                n_remote_nets[k] = n_remote_nets_k
                cst[k] = method.mdl(n_vertices_k, n_pins_k, n_local_nets_k, n_remote_nets_k, k)
            end
        else
            for k = 1:K
                i = Π.spl[k]
                i′ = Π.spl[k + 1]
                n_vertices_k = i′ - i
                n_pins_k = adj_A.colptr[i′] - adj_A.colptr[i]
                n_local_nets_k = 0
                n_remote_nets_k = adj_net[i, i′]
                n_vertices[k] = n_vertices_k
                n_pins[k] = n_pins_k
                n_local_nets[k] = n_local_nets_k
                n_remote_nets[k] = n_remote_nets_k
                cst[k] = method.mdl(n_vertices_k, n_pins_k, n_local_nets_k, n_remote_nets_k, k)
            end
        end

        Π = convert(MapPartition, Π)
        asg = ones(Ti, n)

        for j = randperm(n)
            best_c = typemin(cost_type(method.mdl))
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
                n_local_nets[best_k] += 1
                n_remote_nets[best_k] -= 1
                cst[best_k] = method.mdl(n_vertices[best_k], n_pins[best_k], n_local_nets[best_k], n_remote_nets[best_k], best_k)
            end
        end

        return MapPartition(K, asg)
    end
end