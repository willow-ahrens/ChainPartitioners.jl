using SparseArrays
using LinearAlgebra
using MatrixDepot

matrices = [
    "LPnetlib/lpi_itest6",
    "HB/west0132",
    "LPnetlib/lp_etamacro",
    "LPnetlib/lp_blend",
    "Pajek/GD99_c",
    "HB/can_292",
]

open("matrices.jl", "w") do f
    println(f, "using SparseArrays")
    println(f, "using LinearAlgebra")
    println(f, "matrices = Dict{String, Any}(")
    for mtx in matrices
        A = mdopen(mtx).A
        if issymmetric(A)
            n, n = size(A)
            I, J, V = findnz(SparseMatrixCSC(A.data))
            println(f, "\"$mtx\" => Symmetric(sparse($I, $J, $V, $n, $n), Symbol(\"$(A.uplo)\")),")
        else
            m, n = size(A)
            I, J, V = findnz(SparseMatrixCSC(A))
            println(f, "\"$mtx\" => sparse($I, $J, $V, $m, $n),")
        end
    end
    println(f, ")")
end
