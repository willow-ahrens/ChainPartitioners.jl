function partwise(A::SparseMatrixCSC{Tv, Ti}, Π::MapPartition{Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)
        N = nnz(A)
        pos = A.colptr
        idx = A.rowval
        val = A.nzval
        K = length(Π)

        Πos = zeros(Ti, K + 1)
        πos = zeros(Ti, K + 1)
        hst = zeros(Ti, K)
        idx′ = undefs(Ti, N)
        val′ = undefs(Tv, N)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                k = Π.asg[i]
                πos[k + 1] += hst[k] != j
                hst[k] = j
                k₀ = k
                Πos[k + 1] += 1
            end
        end

        q = 1
        j′ = 1
        for k = 1:(K + 1)
            (Πos[k], q) = (q, q + Πos[k])
            (πos[k], j′) = (j′, j′ + πos[k])
        end
        n′ = j′ - 1

        pos′ = undefs(Ti, n′ + 1)
        prm = zeros(Ti, n′)
        zero!(hst)

        for j = 1:n
            for q in pos[j] : pos[j + 1] - 1
                i = idx[q]
                k = Π.asg[i]
                q′ = Πos[k + 1]
                idx′[q′] = i
                val′[q′] = val[q]
                Πos[k + 1] = q′ + 1
                if hst[k] != j
                    j′ = πos[k + 1]
                    pos′[j′] = q′
                    prm[j′] = j
                    πos[k + 1] = j′ + 1
                end
                hst[k] = j
            end
        end
        pos′[n′ + 1] = N + 1

        return ((n, K, n′, πos, prm), SparseMatrixCSC(m, n′, pos′, idx′, val′))
    end
end

struct PartwiseCost{Ti <: Integer, Ocl}
    n::Int
    K::Int
    n′::Int
    πos::Vector{Ti}
    prm::Vector{Ti}
    ocl::Ocl
end

Base.size(arg::PartwiseCost) = (arg.n + 1, arg.n + 1, arg.K)

partwisecost!(args...; kwargs...) = partwisecost!(NoHint(), args...; kwargs...)
partwisecost!(::AbstractHint, args...; kwargs...) = @assert false
partwisecost!(hint::AbstractHint, n, K, n′, πos, prm, ocl; kwargs...) =
    PartwiseCost(hint, n, K, n′, πos, prm, ocl; kwargs...)

PartwiseCost(hint::AbstractHint, n, K, n′, πos::Vector{Ti}, prm::Vector{Ti}, ocl::Ocl; kwargs...) where {Ti, Ocl} = 
    PartwiseCost{Ti, Ocl}(hint, n, K, n′, πos, prm, ocl; kwargs...)

PartwiseCost{Ti, Ocl}(hint::AbstractHint, n, K, n′, πos::Vector{Ti}, prm::Vector{Ti}, ocl::Ocl; kwargs...) where {Ti, Ocl} = 
    PartwiseCost{Ti, Ocl}(n, K, n′, πos, prm, ocl)

Base.getindex(arg::PartwiseCost{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti} = arg(j, j′, k)
function (arg::PartwiseCost{Ti})(j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = @view arg.prm[arg.πos[k] : arg.πos[k + 1] - 1]
        rnk_j = arg.πos[k] + searchsortedfirst(tmp, j) - 1
        rnk_j′ = arg.πos[k] + searchsortedfirst(tmp, j′) - 1
        return arg.ocl(rnk_j, rnk_j′)
    end
end

mutable struct PartwiseCostStepper{Ti, Ocl} <: AbstractArray{Ti, 3}
    rnk_j::Ti
    rnk_j′::Ti
    cst::PartwiseCost{Ti, Ocl}
end

Base.size(arg::PartwiseCostStepper) = size(arg.cst)

partwisecost!(hint::StepHint, n, K, n′, πos, prm, ocl; kwargs...) =
    PartwiseCostStepper(hint, n, K, n′, πos, prm, ocl; kwargs...)

PartwiseCostStepper(hint::AbstractHint, n, K, n′, πos::Vector{Ti}, prm::Vector{Ti}, ocl::Ocl; kwargs...) where {Ti, Ocl} = 
    PartwiseCostStepper{Ti, Ocl}(hint, n, K, n′, πos, prm, ocl; kwargs...)

function PartwiseCostStepper{Ti, Ocl}(hint::AbstractHint, n, K, n′, πos::Vector{Ti}, prm::Vector{Ti}, ocl::Ocl; kwargs...) where {Ti, Ocl}
    rnk_j = 0
    rnk_j′ = 0
    cst = PartwiseCost{Ti, Ocl}(hint, n, K, n′, πos::Vector{Ti}, prm::Vector{Ti}, ocl::Ocl; kwargs...)
    return PartwiseCostStepper{Ti, Ocl}(rnk_j, rnk_j′, cst)
end

Base.getindex(arg::PartwiseCostStepper{Ti}, j::Integer, j′::Integer, k::Integer) where {Ti} = arg(j, j′, k)
function (arg::PartwiseCostStepper{Ti})(j::Integer, j′::Integer, k::Integer) where {Ti}
    @inbounds begin
        tmp = @view arg.cst.prm[arg.cst.πos[k] : arg.cst.πos[k + 1] - 1]
        rnk_j = arg.cst.πos[k] + searchsortedfirst(tmp, j) - 1
        arg.rnk_j = rnk_j
        rnk_j′ = arg.cst.πos[k] + searchsortedfirst(tmp, j′) - 1
        arg.rnk_j′ = rnk_j′
        return arg.cst.ocl(rnk_j, rnk_j′)
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Same, _j′::Next, _k::Same) where {Ti, Net <: PartwiseCostStepper{Ti}}
    begin
        arg = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        rnk_j = arg.rnk_j
        rnk_j′ = arg.rnk_j′
        πos = arg.cst.πos
        prm = arg.cst.prm
        n′ = arg.cst.n′

        if rnk_j′ < πos[k + 1] && prm[rnk_j′] < j′
            rnk_j′ += 1
            arg.rnk_j′ = rnk_j′
            return Step(arg.cst.ocl)(Same(rnk_j), Next(rnk_j′))
        else
            return Step(arg.cst.ocl)(Same(rnk_j), Same(rnk_j′))
        end
    end
end

@propagate_inbounds function (stp::Step{Net})(_j::Next, _j′::Same, _k::Same) where {Ti, Net <: PartwiseCostStepper{Ti}}
    begin
        arg = stp.ocl
        j = destep(_j)
        j′ = destep(_j′)
        k = destep(_k)
        rnk_j = arg.rnk_j
        rnk_j′ = arg.rnk_j′
        πos = arg.cst.πos
        prm = arg.cst.prm
        n′ = arg.cst.n′

        if rnk_j < πos[k + 1] && prm[rnk_j] < j
            rnk_j += 1
            arg.rnk_j = rnk_j
            return Step(arg.cst.ocl)(Next(rnk_j), Same(rnk_j′))
        else
            return Step(arg.cst.ocl)(Same(rnk_j), Same(rnk_j′))
        end
    end
end