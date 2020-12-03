module ChainPartitioners

using SparseArrays
using LinearAlgebra
using Suppressor
using Random
using Requires

export SparseSummedArea
export SparseSummedRooks
export SparseCountedArea
export SparseCountedRooks
export areasum
export areacount

export EnvelopeMatrix
export rowenvelope

export SparseCountedRowNet
export SparseCountedLocalRowNet
export SparseCountedLocalColNet
export rownetcount
export localrownetcount
export localcolnetcount

export AbstractModel
export AbstractIncreasingModel
export AbstractDecreasingModel

export AbstractWorkCostModel
export AffineWorkCostModel
export WorkCostOracle

export AbstractNetCostModel
export AffineNetCostModel
export NetCostOracle
export AbstractSymCostModel
export AffineSymCostModel
export SymCostOracle
export AbstractCommCostModel
export AffineCommCostModel
export CommCostOracle
export AbstractLocalCostModel
export AffineLocalCostModel
export LocalCostOracle

export AbstractEnvNetCostModel
export AffineEnvNetCostModel
export EnvNetCostOracle

export bottleneck_value
export total_value
export oracle_stripe
export bound_stripe

export partition_stripe
export partition_plaid
export EquiSplitter
export DynamicBottleneckSplitter
export DynamicTotalSplitter
export BisectIndexBottleneckSplitter
export FlipBisectIndexBottleneckSplitter
export BisectCostBottleneckSplitter
export LazyBisectCostBottleneckSplitter
export FlipBisectCostBottleneckSplitter
export SymmetricPartitioner
export MagneticPartitioner
export GreedyBottleneckPartitioner

export pack_stripe
export pack_plaid
export EquiChunker
export DynamicTotalChunker

export MetisPartitioner
export metis_scheme_recursive_bisection
export metis_scheme_direct_k_way
export metis_objective_edge_cut 
export metis_objective_communication_volume 
export metis_coarsen_random 
export metis_coarsen_sorted_heavy_edge 
export metis_initialize_grow 
export metis_initialize_random 
export metis_initialize_edge 
export metis_initialize_node 
export metis_refine_fm 
export metis_refine_greedy 
export metis_refine_fm_two_sided 
export metis_refine_fm_one_sided 

export KaHyParPartitioner
export kahypar_configuration_communication_volume
export kahypar_configuration_hyperedge_cut

export DisjointPartitioner
export AlternatingPartitioner
export AlternatingNetPartitioner
export PermutingPartitioner

export SplitPartition
export DomainPartition
export MapPartition

export permute_stripe
export permute_plaid
export DomainPermutation
export MapPermutation
export IdentityPermuter
export CuthillMcKeePermuter
export GatesCuthillMcKeePermuter
export KryslCuthillMcKeePermuter
export SpectralPermuter
export MinDegreePermuter
export ColumnMinDegreePermuter
export perm

export pattern

include("util.jl")
include("SparsePrefixMatrices.jl")
include("EnvelopeMatrices.jl")
include("Partitions.jl")
include("SparseColorArrays.jl")
include("Costs.jl")
include("WorkCosts.jl")
include("HyperedgeCosts.jl")
include("EnvelopeCosts.jl")
include("EquiPartitioner.jl")
include("DynamicSplitter.jl")
include("DynamicPacker.jl")
include("BisectIndexBottleneckSplitter.jl")
include("BisectCostBottleneckSplitter.jl")
include("LazyBisectCostBottleneckSplitter.jl")
include("SymmetricPartitioner.jl")
include("AlternatingPartitioner.jl")
include("MagneticPartitioner.jl")
include("Permutations.jl")
include("CuthillMcKeePermuter.jl")
include("PermutingPartitioner.jl")

function __init__()
    @require AMD = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e" include("glue_AMD.jl")
    @require Laplacians = "6f8e5838-0efe-5de0-80a3-5fb4f8dbb1de" include("glue_Laplacians.jl")
    @require Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b" include("glue_Metis.jl")
    @require KaHyPar = "2a6221f6-aa48-11e9-3542-2d9e0ef01880" include("glue_KaHyPar.jl")
    @require CuthillMcKee = "17f17636-5e38-52e3-a803-7ae3aaaf3da9" include("glue_CuthillMcKee.jl")
    @require SymRCM = "286e6d88-80af-4590-acc9-0001b223b9bd" include("glue_SymRCM.jl")
end

end # module
