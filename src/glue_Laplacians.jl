struct SpectralPermuter
    assumesymmetry::Bool
    checksymmetry::Bool
end

SpectralPermuter(; assumesymmetry=false, checksymmetry=true) = SpectralPermuter(assumesymmetry, checksymmetry)

function permute_plaid(A::Symmetric{Tv, SparseMatrixCSC{Tv, Ti}}, ::SpectralPermuter; kwargs...) where {Tv, Ti}
    (n, n) = size(A)
    A = SparseMatrixCSC(A)
    A = SparseMatrixCSC{Float64, Ti}(n, n, A.colptr, A.rowval, ones(nnz(A)))
    for j = 1:n
        A[j, j] = 0.0
    end
    dropzeros!(A)

    x = Laplacians.fiedler(A)[2][:, 1]

    prm = DomainPermutation(sortperm(x))

    return (prm, prm)
end

function permute_plaid(A::SparseMatrixCSC{Tv, Ti}, method::SpectralPermuter; kwargs...) where {Tv, Ti}
    (m, n) = size(A)
    A = SparseMatrixCSC{Float64, Ti}(m, n, copy(A.colptr), copy(A.rowval), ones(nnz(A)))
    if method.assumesymmetry || (method.checksymmetry && issymmetric(pattern(A)))
        for j = 1:n
            A[j, j] = 0.0
        end
        dropzeros!(A)

        x = Laplacians.fiedler(A)[2][:, 1]

        prm = DomainPermutation(sortperm(x))

        return (prm, prm)
    else
        B = hvcat((2, 2), spzeros(m, m), A, adjointpattern(A), spzeros(n, n))
        x = Laplacians.fiedler(B)[2][:, 1]
        prm = sortperm(x)
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