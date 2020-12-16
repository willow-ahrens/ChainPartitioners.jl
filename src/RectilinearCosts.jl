struct BlockComponentCostModel{Tv, R, α_Row, α_Col, β_Row<:Tuple{Vararg{Any, R}}, β_Col<:Tuple{Vararg{Any, R}}}
    u_max::Int
    w_max::Int
    α_row::α_Row
    α_col::α_Col
    β_row::β_Row
    β_col::β_Col
end

function BlockComponentCostModel{Tv}((u_max, w_max), α_row::α_Row, α_col::α_Col, β_row::β_Row, β_col::β_Col) where {Tv, R, α_Row, α_Col, β_Row<:Tuple{Vararg{Any, R}}, β_Col<:Tuple{Vararg{Any, R}}}
    return BlockComponentCostModel{Tv, R, α_Row, α_Col, β_Row, β_Col}(u_max, w_max, α_row, α_col, β_row, β_col)
end

@inline cost_type(::Type{<:BlockComponentCostModel{Tv}}) where {Tv} = Tv

struct BlockComponentCostOracle{Tv, Mdl<:BlockComponentCostModel{Tv}} <: AbstractOracleCost{Mdl}
    cst::Matrix{Tv}
    mdl::Mdl
end

@inline (ocl::BlockComponentCostOracle)(j, j′, k) = ocl(j, j′)
@inline function (ocl::BlockComponentCostOracle{Tv, Mdl})(j, j′) where {Tv, Mdl}
    begin
        w = j′ - j
        return ocl.cst[w, j]
    end
end

@propagate_inbounds block_component(f::F, w) where {F} = f(w)
@propagate_inbounds block_component(f::Number, w) = f
@propagate_inbounds block_component(f::Tuple, w) = f[w]
@propagate_inbounds block_component(f::AbstractArray, w) = f[w]

function oracle_stripe(mdl::BlockComponentCostModel{Tc, R}, A::SparseMatrixCSC{Ti, Tv}, Π; net=nothing, adj_A=nothing, kwargs...) where {Ti, Tv, Tc, R}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        u_max = mdl.u_max
        w_max = mdl.w_max
        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        hst = fill(n + 1, K)
        Δ = zeros(Tc, R, n + 1)
        d = zeros(Tc, R)
        cst = zeros(Tc, w_max, n)
        for j = n:-1:1
            for q = A_pos[j] : A_pos[j + 1] - 1
                i = A_idx[q]
                k = Π_asg[i]
                u = Π_spl[k + 1] - Π_spl[k]
                if hst[k] > j
                    for r = 1:R
                        Δ[r, hst[k]] -= block_component(mdl.β_row[r], u)
                    end
                    for r = 1:R
                        Δ[r, j] += block_component(mdl.β_row[r], u)
                    end
                end
                hst[k] = j
            end
            zero!(d)
            for j′ = j + 1:min(j + w_max, n + 1)
                w = j′ - j
                for r = 1:R
                    d[r] += Δ[r, j′ - 1] 
                end
                c = zero(Tv)
                for r = 1:R
                    c += d[r] * block_component(mdl.β_col[r], w)
                end
                cst[w, j] = c + block_component(mdl.α_col, w)
            end
        end

        return BlockComponentCostOracle(cst, mdl)
    end
end

function total_value(A::SparseMatrixCSC, Π, Φ, mdl::BlockComponentCostModel{Tc, R}) where {Tc, R}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        Φ = convert(DomainPartition, Φ)
        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        L = length(Φ)
        hst = zeros(Int, K)
        c = zero(Tc)
        for l = 1:L
            w = Φ.spl[l + 1] - Φ.spl[l]
            c += block_component(mdl.α_col, w)
            for _j = Φ.spl[l] : Φ.spl[l + 1] - 1
                j = Φ.prm[_j]
                for q = A_pos[j] : A_pos[j + 1] - 1
                    i = A_idx[q]
                    k = Π_asg[i]
                    u = Π_spl[k + 1] - Π_spl[k]
                    if hst[k] < l
                        for r = 1:R
                            c += block_component(mdl.β_row[r], u) * block_component(mdl.β_col[r], w)
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
total_partition_value(Π, mdl::BlockComponentCostModel)
    c_α = zero(Tv)
    for k = 1:K
        u = Π_spl[k + 1] - Π_spl[k]
        c_α += block_component(mdl.α_row, u)
    end
    return c_α
end
=#