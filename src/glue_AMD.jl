struct MinDegreePermuter
    assumesymmetry::Bool
    checksymmetry::Bool
end

MinDegreePermuter(; assumesymmetry=false, checksymmetry=true) = MinDegreePermuter(assumesymmetry, checksymmetry)

function permute_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::MinDegreePermuter) where {Tv, Ti}
    prm = AMD.symamd(SparseMatrixCSC(A))
    return DomainPermutation(prm)
end

function permute_stripe(A::SparseMatrixCSC, method::MinDegreePermuter)
    return permute_plaid(A, method)[end]
end

function permute_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::MinDegreePermuter) where {Tv, Ti}
    prm = AMD.symamd(SparseMatrixCSC(A))
    return (DomainPermutation(prm), DomainPermutation(prm))
end

function permute_plaid(A::SparseMatrixCSC, method::MinDegreePermuter)
    m, n = size(A)
    if m == n
        if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
            prm = AMD.symamd(A)
            return (DomainPermutation(prm), DomainPermutation(prm))
        else
            prm = AMD.amd(A)
            return (DomainPermutation(prm), DomainPermutation(prm))
        end
    else
        B = hvcat((2, 2), spzeros(m, m), A, spzeros(n, m), spzeros(n, n))
        prm = AMD.amd(B)
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

struct ColumnMinDegreePermuter
    assumesymmetry::Bool
    checksymmetry::Bool
end

ColumnMinDegreePermuter(; assumesymmetry=false, checksymmetry=true) = ColumnMinDegreePermuter(assumesymmetry, checksymmetry)

function permute_stripe(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, method::ColumnMinDegreePermuter) where {Tv, Ti}
    prm = AMD.colamd(SparseMatrixCSC(A))
    return DomainPermutation(prm)
end

function permute_stripe(A::SparseMatrixCSC, method::ColumnMinDegreePermuter)
    prm = AMD.colamd(A)
    return DomainPermutation(prm)
end