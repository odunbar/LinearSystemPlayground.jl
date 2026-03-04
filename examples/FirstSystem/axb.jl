# build matrix pkg
using LinearAlgebra
using SparseArrays
using Random

#solver pkg
using IterativeSolvers
using IncompleteLU

#benchmark pkg
using BenchmarkTools

Random.seed!(153452)
n = 500 # size of blocks

include("build_a.jl") # builds the matri
include("build_rhs.jl") # builds the rhs

# build the linear systems

A, T1, U, Vt, T2 = build_a(n)
b = build_rhs(size(A,2))

# Full solves

# [1] Naive solve
@btime $A \ $b

# [2] Iterative solve - no precond
@btime gmres($A, $b)

# [3] Iterative solve - with basic left precond
P = ilu(A)              # ILU(0)
gmres(A, b, Pl=P)  # left preconditioning

@btime gmres($A, $b, Pl=$P)

# [4] Jacobi solve
k=20
Ad = diag(A)
iAd = 1 ./ Ad
x_it = zeros(size(A,2))
for i=1:k
    x_it .= x_it + iAd.*(b - A*x_it)
    println(norm(x_it-x))
end
# I wont time as this iteration blows up


