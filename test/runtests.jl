using Test
using ChainPartitioners
using SparseArrays
using Random
include("matrices.jl")

macro ignore(ex)
    nothing
end

@testset "ChainPartitioners" begin
    Random.seed!(0xDEADBEEF)
    include("test_util.jl")
    include("test_SparsePrefixMatrices.jl")
    include("test_EnvelopeMatrices.jl")
    include("test_Partitions.jl")
    include("test_SparseColorArrays.jl")
    include("test_Costs.jl")
    include("test_Partitioners.jl")
    include("test_CuthillMcKee.jl")
end
