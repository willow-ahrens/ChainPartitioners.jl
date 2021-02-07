struct ColumnBlockComponentCostModel{Tv, α_Col, β_Col} <: AbstractNetCostModel
    W::Int
    α_col::α_Col
    β_col::β_Col
end

function ColumnBlockComponentCostModel{Tv}(W, α_col::α_Col, β_col::β_Col) where {Tv, α_Col, β_Col}
    return ColumnBlockComponentCostModel{Tv, α_Col, β_Col}(W, α_col, β_col)
end

@inline cost_type(::Type{<:ColumnBlockComponentCostModel{Tv}}) where {Tv} = Tv

(mdl::ColumnBlockComponentCostModel{Tv, α_Col, β_Col})(x_width, x_work, x_net) where {Tv, α_Col, β_Col} = block_component(mdl.α_col, x_width) + x_net * block_component(mdl.β_col, x_width)

struct BlockComponentCostModel{Tv, R, α_Row, α_Col, β_Row<:Tuple{Vararg{Any, R}}, β_Col<:Tuple{Vararg{Any, R}}}
    U::Int
    W::Int
    α_row::α_Row
    α_col::α_Col
    β_row::β_Row
    β_col::β_Col
end

function Base.permutedims(cst::BlockComponentCostModel{Tv}) where {Tv}
    return BlockComponentCostModel{Tv}(cst.W, cst.U, cst.α_col, cst.α_row, cst.β_col, cst.β_row)
end

function BlockComponentCostModel{Tv}(U, W, α_row::α_Row, α_col::α_Col, β_row::β_Row, β_col::β_Col) where {Tv, R, α_Row, α_Col, β_Row<:Tuple{Vararg{Any, R}}, β_Col<:Tuple{Vararg{Any, R}}}
    return BlockComponentCostModel{Tv, R, α_Row, α_Col, β_Row, β_Col}(U, W, α_row, α_col, β_row, β_col)
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

function oracle_stripe(hint::AbstractHint, mdl::BlockComponentCostModel{Tc, R}, A::SparseMatrixCSC{Ti, Tv}, Π; net=nothing, adj_A=nothing, kwargs...) where {Ti, Tv, Tc, R}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        U = mdl.U
        W = mdl.W
        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        hst = fill(n + 1, K)
        Δ = zeros(Tc, R, n + 1)
        d = zeros(Tc, R)
        cst = zeros(Tc, W, n)
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
            for j′ = j + 1:min(j + W, n + 1)
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
mutable struct BlockComponentCostStepOracle{Tv, Ti, Tc, Mdl<:BlockComponentCostModel{Tc}} <: AbstractOracleCost{Mdl}
    A::SparseMatrixCSC{Tv, Ti}
    Π_asg::MapPartition{Ti}
    Π_spl::SplitPartition{Ti}
    mdl::Mdl
    hst::Vector{Ti}
    Δ::Matrix{Tc}
    d::Matrix{Tc}
    j::Ti
    j′::Ti
end

function oracle_stripe(hint::StepHint, mdl::BlockComponentCostModel{Tc, R}, A::SparseMatrixCSC{Tv, Ti}, Π; kwargs...) where {Tv, Ti, Tc, R}
    @inbounds begin
        m, n = size(A)
        K = length(Π)
        return BlockComponentCostStepOracle(A, convert(MapPartition, Π), mdl, ones(Ti, K), undefs(Ti, n + 1), zeros(Tc, R), Ti(1), Ti(1))
    end
end

oracle_model(ocl::BlockComponentCostStepOracle) = ocl.mdl

@inline function (ocl::BlockComponentCostStepOracle{Tv, Ti, Mdl})(j::Ti, j′::Ti, k...) where {Tv, Ti, Mdl}
    @inbounds begin
        ocl_j = ocl.j
        ocl_j′ = ocl.j′
        q′ = ocl.q′
        A = ocl.A
        Π = ocl.Π
        d = ocl.d
        Δ = ocl.Δ
        pos = A.colptr
        idx = A.rowval
        hst = ocl.hst

        if j′ < ocl_j′
            ocl_j = Ti(1)
            ocl_j′ = Ti(1)
            zero!(d)
            one!(ocl.hst)
        end
        while ocl_j′ < j′
            q = q′
            q′ = pos[ocl_j′ + 1]

            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                k = Π_asg[i]
                u = Π_spl[k + 1] - Π_spl[k]
                if hst[k] < j′
                    for r = 1:R
                        Δ[r, hst[k]] -= block_component(f.β_row[r], u)
                    end
                    for r = 1:R
                        Δ[r, j′ - 1] += block_component(f.β_row[r], u)
                    end
                end
                hst[k] = ocl_j′ + 1


            Δ_net[ocl_j′ + 1] = q′ - q
            for _q = q:q′ - 1
                i = idx[_q]
                j₀ = hst[i] - 1
                x_net += j₀ < ocl_j
                Δ_net[j₀ + 1] -= 1
                hst[i] = ocl_j′ + 1
            end
            ocl_j′ += 1
        end
        if j == j′ - 1
            ocl_j = j′ - 1
            x_net = Δ_net[ocl_j + 1]
        elseif j == j′
            ocl_j = j′
            x_net = Ti(0)
        else
            while j < ocl_j
                ocl_j -= 1
                x_net += Δ_net[ocl_j + 1]
            end
            while j > ocl_j
                x_net -= Δ_net[ocl_j + 1]
                ocl_j += 1
            end
        end

        ocl.j = ocl_j
        ocl.j′ = ocl_j′
        ocl.q′ = q′
        ocl.x_net = x_net
        return ocl.mdl(j′ - j, q′ - pos[j], x_net, k...)
    end
end
=#

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

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{F}, args...; kwargs...) where {F<:BlockComponentCostModel, Tv, Ti}
    return pack_stripe(A, DynamicTotalChunker(ConstrainedCost(method.f, FeasibleCost(), Feasible())), args..., kwargs...)
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{<:ConstrainedCost{F}}, Π; kwargs...) where {Tc, R, F<:BlockComponentCostModel{Tc, R}, Tv, Ti}
    @inbounds begin
        m, n = size(A)
        A_pos = A.colptr
        A_idx = A.rowval

        f = method.f.f
        w = oracle_stripe(StepHint(), method.f.w, A, Π) #TODO reverse outer loop so step oracles can have stationary j′
        w_max = method.f.w_max
        Π_asg = convert(MapPartition, Π).asg
        Π_spl = convert(DomainPartition, Π).spl
        K = length(Π)
        hst = fill(n + 1, K)
        Δ = zeros(Tc, R, n + 1)
        d = zeros(Tc, R)
        cst = Vector{cost_type(f)}(undef, n + 1) # cst[j] is the best cost of a partition from j to n
        spl = Vector{Int}(undef, n + 1)
        for j = n:-1:1
            for q = A_pos[j] : A_pos[j + 1] - 1
                i = A_idx[q]
                k = Π_asg[i]
                u = Π_spl[k + 1] - Π_spl[k]
                if hst[k] > j
                    for r = 1:R
                        Δ[r, hst[k]] -= block_component(f.β_row[r], u)
                    end
                    for r = 1:R
                        Δ[r, j] += block_component(f.β_row[r], u)
                    end
                end
                hst[k] = j
            end
            zero!(d)
            for r = 1:R
                d[r] += Δ[r, j] 
            end
            best_c = cst[j + 1]
            for r = 1:R
                best_c += d[r] * block_component(f.β_col[r], 1)
            end
            best_c += block_component(f.α_col, 1)
            best_j′ = j + 1
            for j′ = j + 2: n + 1
                println((w(j, j′), j′ - j))
                if w(j, j′) > w_max
                    break
                end
                for r = 1:R
                    d[r] += Δ[r, j′ - 1] 
                end
                c = cst[j′]
                for r = 1:R
                    c += d[r] * block_component(f.β_col[r], j′ - j)
                end
                c += block_component(f.α_col, j′ - j)
                if c < best_c
                    best_c = c
                    best_j′ = j′
                end
            end
            cst[j] = best_c
            spl[j] = best_j′
        end

        K = 0
        j = 1
        while j != n + 1
            j′ = spl[j]
            K += 1
            spl[K] = j
            j = j′
        end
        spl[K + 1] = j
        resize!(spl, K + 1)
        return SplitPartition{Ti}(K, spl)
    end
end