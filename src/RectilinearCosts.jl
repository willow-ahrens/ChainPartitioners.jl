struct BlockBasisCostModel{R, α_Row, α_Col, β_Row<:Tuple{Vararg{Any, R}}, β_Col<:Tuple{Vararg{Any, R}}}
    α_row::α_Row
    α_col::α_Col
    β_row::β_Row
    β_col::β_Col
end

@inline cost_type(::Type{BlockBasisCostModel{Tv, F}}) where {Tv, F} = #WTF goes here? #promote_type(Tv, cost_type(F))

struct BlockBasisCostOracle{Tv, Mdl<:BlockBasisCostModel} <: AbstractOracleCost{Mdl}
    cst::Matrix{Tv}
    mdl::Mdl
end

@inline cost_type(::Type{BlockBasisCostOracle{Tv}}) where {Tv} = Tv

@propagate_inbounds block_basis(f::F, w) where {F} = f(w)
@propagate_inbounds block_basis(f::AbstractNumber, w) = f
@propagate_inbounds block_basis(f::Tuple, w) = f[w]
@propagate_inbounds block_basis(f::AbstractArray, w) = f[w]

function oracle_stripe(mdl::BlockBasisCostModel{R}, A::SparseMatrixCSC{Ti, Tv}, Π; net=nothing, adj_A=nothing, kwargs...) where {Ti, Tv}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        hst = fill(n + 1, K)
        Tc = cost_type(mdl)
        Δ = zeros(Tc, R, n + 1)
        d = zeros(Tc, R)
        cst_β = zeros(Tc, w_max, n)
        for j = n:-1:1
            for q = A_pos[j] : A_pos[j + 1] - 1
                i = A_idx[q]
                k = Π_asg[i]
                u = Π_spl[k + 1] - Π_spl[k]
                if hst[k] > j
                    for r = 1:R
                        Δ[r, hst[k]] -= block_basis(mdl.β_row[r], u)
                    end
                    for r = 1:R
                        Δ[r, j] += block_basis(mdl.β_row[r], u)
                    end
                end
                hst[k] = j
            end
            zero!(d)
            for j′ = j + 1:min(j + w_max + 1, n + 1)
                w = j′ - j
                for r = 1:R
                    d[r] += Δ[r, j] 
                end
                c = zero(Tv)
                for r = 1:R
                    c += d[r] * block_basis(mdl.β_col[r], w)
                end
                cst[j, w] = c + block_basis(mdl.α_col, w)
            end
        end

        return BlockBasisOracle(cst_β, mdl)
    end
end

function total_value(A::SparseMatrixCSC, Π, Φ::DomainPartition, mdl::BlockBasisCostModel) where {G}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        L = length(Φ)
        hst = fill(n + 1, K)
        Tc = cost_type(mdl)
        Δ = zeros(Tc, R, n + 1)
        d = zeros(Tc, R)
        cst_β = zeros(Tc, w_max, n)
        for l = 1:L
            w = Φ.spl[l + 1] - Φ.spl[l]
            for _j = Φ.spl[l] : Φ.spl[l + 1] - 1
                j = Φ.prm[_j]
                for q = A_pos[j] : A_pos[j + 1] - 1
                    i = A_idx[q]
                    k = Π_asg[i]
                    u = Π_spl[k + 1] - Π_spl[k]
                    if hst[k] > l
                        for r = 1:R
                            c += block_basis(mdl.β_row[r], u) * block_basis(mdl.β_col[r], w)
                        end
                    end
                    hst[k] = l
                end
            end
        end

        return c
    end
end

#=
total_partition_value(Π, mdl::BlockBasisCostModel)
    c_α = zero(Tv)
    for k = 1:K
        u = Π_spl[k + 1] - Π_spl[k]
        c_α += block_basis(mdl.α_row, u)
    end
    return c_α
end
=#