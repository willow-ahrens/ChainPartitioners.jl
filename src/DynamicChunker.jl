abstract type AbstractDynamicChunker{F} end

struct DynamicBottleneckChunker{F} <: AbstractDynamicChunker{F}
    f::F
end

@inline _dynamic_chunker_combine(::DynamicBottleneckChunker) = max

struct DynamicTotalChunker{F} <: AbstractDynamicChunker{F}
    f::F
end

@inline _dynamic_chunker_combine(::DynamicTotalChunker) = +

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker, args...; kwargs...) where {Tv, Ti}
    #Reference Implementation
    return pack_stripe(A, DynamicTotalChunker(ConstrainedCost(method.f, FeasibleCost(), Feasible())), args..., kwargs...)
end

function pack_stripe(A::SparseMatrixCSC{Tv, Ti}, method::DynamicTotalChunker{<:ConstrainedCost}, args...; kwargs...) where {Tv, Ti}
    #Reference Implementation
    @inbounds begin
        # matrix notation...
        # i = 1:m rows, j = 1:n columns
        m, n = size(A)

        f = oracle_stripe(StepHint(), method.f.f, A, args...)
        w = oracle_stripe(StepHint(), method.f.w, A, args...)
        w_max = method.f.w_max

        cst = Vector{cost_type(f)}(undef, n + 1)
        cst[1] = zero(cost_type(f))
        spl = Vector{Int}(undef, n + 1)
        j₀ = 1
        for j′ = 2:n + 1
            while w(j₀, j′) > w_max
                j₀ += 1
            end
            @assert j₀ < j′ #TODO infeasibility

            best_c = cst[j₀] + f(j₀, j′)
            best_j = j₀
            for j = j₀ + 1 : j′ - 1
                c = cst[j] + f(j, j′) 
                if c < best_c
                    best_c = c
                    best_j = j
                end
            end
            cst[j′] = best_c
            spl[j′] = best_j
        end

        return unravel_chunks!(spl, n)
    end
end

function unravel_chunks!(spl, n)
    begin
        K = 0
        j′ = n + 1
        while j′ != 1
            j = spl[j′]
            spl[end - K] = j′
            K += 1
            j′ = j
        end
        spl[1] = 1
        for k = 1:K
            spl[k + 1] = spl[end - K + k]
        end
        resize!(spl, K + 1)
        return SplitPartition(K, spl)
    end
end