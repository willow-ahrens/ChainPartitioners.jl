struct GatesCuthillMcKeePermuter
    rev::Bool
    warnunconnected::Bool
    assumesymmetry::Bool
    checksymmetry::Bool
end

GatesCuthillMcKeePermuter(; rev=true, warnunconnected=false, assumesymmetry=false, checksymmetry=true) = GatesCuthillMcKeePermuter(rev, warnunconnected, assumesymmetry, checksymmetry)

function permute_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::GatesCuthillMcKeePermuter; kwargs...) where {Tv, Ti}
    prm = CuthillMcKee.symrcm(SparseMatrixCSC(A), method.rev, false, method.warnunconnected)
    return DomainPermutation(prm)
end

function permute_stripe(A::SparseMatrixCSC, method::GatesCuthillMcKeePermuter; kwargs...)
    return permute_plaid(A, method; kwargs...)[end]
end

function permute_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::GatesCuthillMcKeePermuter; kwargs...) where {Tv, Ti}
    prm = CuthillMcKee.symrcm(SparseMatrixCSC(A), method.rev, false, method.warnunconnected)
    return (DomainPermutation(prm), DomainPermutation(prm))
end

function permute_plaid(A::SparseMatrixCSC, method::GatesCuthillMcKeePermuter; adj_A = nothing, kwargs...)
    if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
        prm = CuthillMcKee.symrcm(A, method.rev, false, method.warnunconnected)
        return (DomainPermutation(prm), DomainPermutation(prm))
    else
        if adj_A === nothing
            adj_A = adjointpattern(A)
        end
        m, n = size(A)
        B = hvcat((2, 2), spzeros(m, m), A, adj_A, spzeros(n, n))
        prm = CuthillMcKee.symrcm(B, method.rev, false, method.warnunconnected)
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