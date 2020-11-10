function bandwidth(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    bot = zeros(Ti, n + 1)
    bot[n + 1] = m + 1
    for j = n:-1:1
        if A.colptr[j] < A.colptr[j + 1]
            bot[j] = min(bot[j + 1], A.rowval[A.colptr[j]])
        else
            bot[j] = bot[j + 1]
        end
    end
    top = zeros(Ti, n + 1)
    top[1] = 0
    for j = 1:n
        if A.colptr[j] < A.colptr[j + 1]
            top[j + 1] = max(top[j], A.rowval[A.colptr[j + 1] - 1])
        else
            top[j + 1] = top[j]
        end
    end

    return maximum(top[1:n] .- bot[2:n + 1])
end

@testset "CuthillMcKee" begin
    A = matrices["HB/can_292"]
    (σ, τ) = permute_plaid(A, CuthillMcKeePermuter())
    @test σ.prm === τ.prm
    (σ′, τ′) = permute_plaid(A, CuthillMcKeePermuter(reverse=true))
    @test σ.prm == reverse(σ′.prm)
    @test τ.prm == reverse(τ′.prm)
    A = SparseMatrixCSC(A)
    (σ, τ) = permute_plaid(A, CuthillMcKeePermuter())
    @test σ.prm === τ.prm
    (σ′, τ′) = permute_plaid(A, CuthillMcKeePermuter(reverse=true))
    @test σ.prm == reverse(σ′.prm)
    @test τ.prm == reverse(τ′.prm)
    @test bandwidth(A[perm(σ), perm(τ)]) < bandwidth(A)
    @test bandwidth(permutedims(A[perm(σ), perm(τ)])) < bandwidth(permutedims(A))
    A = matrices["HB/west0132"]
    (σ, τ) = permute_plaid(A, CuthillMcKeePermuter())
    (σ′, τ′) = permute_plaid(A, CuthillMcKeePermuter(reverse=true))
    @test σ.prm == reverse(σ′.prm)
    @test τ.prm == reverse(τ′.prm)
    @test bandwidth(A[perm(σ), perm(τ)]) < bandwidth(A)
    @test bandwidth(permutedims(A[perm(σ), perm(τ)])) < bandwidth(permutedims(A))
    A = matrices["LPnetlib/lp_etamacro"]
    (σ, τ) = permute_plaid(A, CuthillMcKeePermuter())
    (σ′, τ′) = permute_plaid(A, CuthillMcKeePermuter(reverse=true))
    @test σ.prm == reverse(σ′.prm)
    @test τ.prm == reverse(τ′.prm)
    @test bandwidth(A[perm(σ), perm(τ)]) < bandwidth(A)
    @test bandwidth(permutedims(A[perm(σ), perm(τ)])) < bandwidth(permutedims(A))
end
