struct KryslCuthillMcKeePermuter
    sortbydeg::Bool
    assumesymmetry::Bool
    checksymmetry::Bool
end

KryslCuthillMcKeePermuter(; sortbydeg=false, assumesymmetry=false, checksymmetry=true) = KryslCuthillMcKeePermuter(sortbydeg, assumesymmetry, checksymmetry)

function permute_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::KryslCuthillMcKeePermuter; kwargs...) where {Tv, Ti}
    prm = SymRCM.symrcm(SparseMatrixCSC(A), sortbydeg=method.sortbydeg)
    return DomainPermutation(prm)
end

function permute_stripe(A::SparseMatrixCSC, method::KryslCuthillMcKeePermuter; kwargs...)
    return permute_plaid(A, method)[end]
end

function permute_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::KryslCuthillMcKeePermuter; kwargs...) where {Tv, Ti}
    prm = SymRCM.symrcm(SparseMatrixCSC(A), sortbydeg=method.sortbydeg)
    return (DomainPermutation(prm), DomainPermutation(prm))
end

function permute_plaid(A::SparseMatrixCSC, method::KryslCuthillMcKeePermuter; adj_A = nothing, kwargs...)
    if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
        prm = SymRCM.symrcm(A, sortbydeg=method.sortbydeg)
        return (DomainPermutation(prm), DomainPermutation(prm))
    else
        m, n = size(A)
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        B = hvcat((2, 2), spzeros(m, m), A, adj_A, spzeros(n, n))
        prm = SymRCM.symrcm(B, sortbydeg=method.sortbydeg)
        iprm = Int[]
        jprm = Int[]
        for v in prm
            if v > m
                push!(jprm, v - m)
            else
                push!(iprm, v)
            end
        end
        return (DomainPermutation(iprm), DomainPermutation(jprm))
    end
end