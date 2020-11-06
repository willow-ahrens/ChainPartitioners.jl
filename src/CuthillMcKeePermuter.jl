struct CuthillMcKeePermuter
    reverse::Bool
    assumesymmetry::Bool
    checksymmetry::Bool
end

function symrcm(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)

        deg = undefs(Ti, max(m + 2, n))
        bkt = undefs(Ti, n)
        ord = undefs(Ti, n)
        deg[1] = 1
        for i = 2:m + 2
            deg[i] = 0
        end
        for j = 1:n
            deg[A.colptr[j + 1] - A.colptr[j] + 2] += 1
        end
        for i = 1:m + 1
            deg[i + 1] += deg[i]
        end
        for j = 1:n
            d = A.colptr[j + 1] - A.colptr[j] + 1
            k = deg[d]
            ord[k] = j
            deg[d] = k + 1
        end

        k_start = 1

        prm = undefs(Ti, n)
        vst = falses(n)
        k_current = 0
        k_frontier = k_current
        while k_frontier < n
            k_current += 1
            if k_current > k_frontier
                while vst[ord[k_start]]
                    k_start += 1
                end
                j = ord[k_start]
                prm[k_current] = j
                k_frontier = k_current
                vst[j] = true
            else
                j = prm[k_current]
            end
            k_frontier′ = k_frontier
            for q = A.colptr[j]:A.colptr[j + 1] - 1
                i = A.rowval[q]
                if !vst[i]
                    k_frontier′ += 1
                    prm[k_frontier′] = i
                    vst[i] = true
                end
            end

            if k_frontier′ > k_frontier
                let i = prm[k_frontier′]
                    deg[k_frontier′] = A.colptr[i + 1] - A.colptr[i]
                    bkt[k_frontier′] = k_frontier′
                end
                for k = k_frontier′-1:-1:k_frontier + 1
                    i = prm[k]
                    d = A.colptr[i + 1] - A.colptr[i]
                    while k != k_frontier′ && d > deg[k + 1]
                        deg[k] = deg[bkt[k + 1]]
                        prm[k] = prm[bkt[k + 1]]
                        bkt[k] = bkt[k + 1] - (d != deg[k + 1])
                        k = bkt[k + 1]
                    end
                    prm[k] = i
                    deg[k] = d
                    bkt[k] = k
                end

                k_frontier = k_frontier′
            end
        end
        return prm
    end
end

function rctrcm(A::SparseMatrixCSC{Tv, Ti}, adj_A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)

        deg = undefs(Ti, max(m + 2, n + 2, m + n))
        bkt = undefs(Ti, m + n)
        ord = undefs(Ti, m + n)
        deg[1] = 1
        for v = 2:max(n + 2, m + 2)
            deg[v] = 0
        end
        for i = 1:m
            deg[adj_A.colptr[i + 1] - adj_A.colptr[i] + 2] += 1
        end
        for j = 1:n
            deg[A.colptr[j + 1] - A.colptr[j] + 2] += 1
        end
        for v = 1:max(m + 1, n + 1)
            deg[v + 1] += deg[v]
        end
        for i = 1:m
            d = adj_A.colptr[i + 1] - adj_A.colptr[i] + 1
            k = deg[d]
            ord[k] = i
            deg[d] = k + 1
        end
        for j = 1:n
            d = A.colptr[j + 1] - A.colptr[j] + 1
            k = deg[d]
            ord[k] = m + j
            deg[d] = k + 1
        end

        k_start = 1

        prm = undefs(Ti, m + n)
        vst = falses(m + n)
        k_current = 0
        k_frontier = k_current
        while k_frontier < m + n
            k_current += 1
            if k_current > k_frontier
                while vst[ord[k_start]]
                    k_start += 1
                end
                v = ord[k_start]
                prm[k_current] = v
                k_frontier = k_current
                vst[v] = true
            else
                v = prm[k_current]
            end
            k_frontier′ = k_frontier
            if v <= m
                i = v
                for q = adj_A.colptr[i]:adj_A.colptr[i + 1] - 1
                    j = adj_A.rowval[q]
                    if !vst[m + j]
                        k_frontier′ += 1
                        prm[k_frontier′] = m + j
                        vst[m + j] = true
                    end
                end
            else
                j = v - m
                for q = A.colptr[j]:A.colptr[j + 1] - 1
                    i = A.rowval[q]
                    if !vst[i]
                        k_frontier′ += 1
                        prm[k_frontier′] = i
                        vst[i] = true
                    end
                end
            end

            if k_frontier′ > k_frontier
                let v = prm[k_frontier′]
                    if v <= m
                        i = v
                        deg[k_frontier′] = adj_A.colptr[i + 1] - adj_A.colptr[i]
                    else
                        j = v - m
                        deg[k_frontier′] = A.colptr[j + 1] - A.colptr[j]
                    end
                    bkt[k_frontier′] = k_frontier′
                end
                for k = k_frontier′-1:-1:k_frontier + 1
                    v = prm[k]
                    if v <= m
                        i = v
                        d = adj_A.colptr[i + 1] - adj_A.colptr[i]
                    else
                        j = v - m
                        d = A.colptr[j + 1] - A.colptr[j]
                    end
                    while k != k_frontier′ && d > deg[k + 1]
                        deg[k] = deg[bkt[k + 1]]
                        prm[k] = prm[bkt[k + 1]]
                        bkt[k] = bkt[k + 1] - (d != deg[k + 1])
                        k = bkt[k + 1]
                    end
                    prm[k] = v
                    deg[k] = d
                    bkt[k] = k
                end

                k_frontier = k_frontier′
            end
        end
        return prm
    end
end

CuthillMcKeePermuter(; reverse=false, assumesymmetry=false, checksymmetry=true) = CuthillMcKeePermuter(reverse, assumesymmetry, checksymmetry)

function permute_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::CuthillMcKeePermuter) where {Tv, Ti}
    prm = symrcm(SparseMatrixCSC(A))
    if method.reverse
        reverse!(prm)
    end
    return DomainPermutation(prm)
end

function permute_stripe(A::SparseMatrixCSC, method::CuthillMcKeePermuter)
    return permute_plaid(A, method)[end]
end

function permute_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::CuthillMcKeePermuter) where {Tv, Ti}
    prm = symrcm(SparseMatrixCSC(A))
    if method.reverse
        reverse!(prm)
    end
    return (DomainPermutation(prm), DomainPermutation(prm))
end

function permute_plaid(A::SparseMatrixCSC{Tv, Ti}, method::CuthillMcKeePermuter; adj_A = nothing) where {Tv, Ti}
    @inbounds begin
        if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
            prm = symrcm(A)
            if method.reverse
                reverse!(prm)
            end
            return (DomainPermutation(prm), DomainPermutation(prm))
        else
            m, n = size(A)
            if adj_A === nothing
                adj_A = adjointpattern(A)
            end
            prm = rctrcm(A, adj_A)
            iprm = undefs(Ti, m)
            jprm = undefs(Ti, n)
            if method.reverse
                i = m
                j = n
                for v in prm
                    if v <= m
                        iprm[i] = v
                        i -= 1
                    else
                        jprm[j] = v - m
                        j -= 1
                    end
                end
            else
                i = 1
                j = 1
                for v in prm
                    if v <= m
                        iprm[i] = v
                        i += 1
                    else
                        jprm[j] = v - m
                        j += 1
                    end
                end
            end
            return (DomainPermutation(iprm), DomainPermutation(jprm))
        end
    end
end