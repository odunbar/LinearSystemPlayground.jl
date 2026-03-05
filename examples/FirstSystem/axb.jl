# build matrix pkg
using LinearAlgebra
using LinearMaps
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

# -----------
# Full solves: A
# ----------- 

# [1] Naive solve
@btime $A \ $b
@allocated A \ b
# [2] Iterative solve - no precond
@btime gmres($A, $b)
@allocated gmres(A,b)

# [3] Iterative solve - with basic left precond
P = ilu(A)              # ILU(0)
@btime gmres($A, $b, Pl=$P)  # left preconditioning
@allocated gmres(A,b, Pl=P)


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

# ------------
# Schur solves: S & T1
# ------------

# split rhs
s1 = size(T1,1) 

b1, b2 = b[1:s1], b[s1+1:end]

# for btime, preallocate sizes:
Umat = Matrix(U)

# [1] Naive solve
@btime begin
    z_b = $T1 \ $b1
    z_U = $T1 \ $Umat
    S2 = $T2 - $Vt * z_U 
    x2 = $S2 \ ($b2 - $Vt*z_b)
    x1 = z_b - z_U * x2
end

# [2] GMRES with no precond, and allocates

# define the linear map:
schur_action(x) = T2 * x - Vt * (T1 \ (U*x))
schur_adj_action(x) = T2' * x - U' * (T1 \ (Vt'*x))

S2 = LinearMap(
    schur_action,
    schur_adj_action,
    size(T2,1),
    size(T2,1),
)

@btime begin
    rhs = $b2 - $Vt*($T1\$b1)
    x2 = gmres($S2, rhs)
    x1 = $T1 \ ($b1-$Umat*x2)
end


# [2] GMRES with no precond, no allocates
tmp1 = zeros(eltype(T1), size(U,1))
tmp2 = similar(tmp1)
F = factorize(Matrix(T1))

function schur_action!(y,x)
    mul!(tmp1, Umat, x)
    ldiv!(tmp2, F, tmp1) # tmp2=T1\(U*x)
    mul!(y, T2, x) # y= T2*x
    mul!(y, Vt, tmp2, -1, 1) # y = 1 y - 1 V'* tmp2
end

tmp3 = zeros(eltype(T1), size(Vt,2))
tmp4 = similar(tmp3)
function schur_adj_action!(y,x)
    mul!(tmp3, Vt', x)
    ldiv!(tmp4, F', tmp3) # tmp4 = T1'\(V*x)
    mul!(y, T2', x) # y= T2'*x
    mul!(y, U', tmp4, -1, 1) # y = 1 y - 1 U'* tmp4
end

S2_noalloc = LinearMap(
    schur_action!,
    schur_adj_action!,
    size(T2,1),
    size(T2,1),
)
@btime begin
    rhs = $b2 - $Vt*($F\$b1)
    x2 = gmres($S2_noalloc, rhs)
    x1 = $F \ ($b1-$Umat*x2)
end

# [2] GMRES with precond, no allocates
# Here is a reasonable preconditioner:
P = ilu(T2)              # ILU(0)
@btime begin
    rhs = $b2 - $Vt*($F\$b1)
    x2 = gmres($S2_noalloc, rhs, Pl=$P)
    x1 = $Ff \ ($b1-$Umat*x2)
end


# lesson learnt, don't do GMRES on the schur complement due to the internal linear solve (and poorer conditioning)

# [3] Jacobi
#true soln
x_true = A\b
x1_true,x2_true = x_true[1:s1], x_true[s1+1:end]
    
k=20
Td = diag(T1)
iTd = 1 ./ Td

z_U = T1 \ Umat
z_b = T1 \ b1
rhs = b2-Vt*z_b
S2 = Matrix(T2) - Vt * z_U

x_it = zeros(size(T2,1))

# correct Jacobi preconditioning
iS2 = 1 ./ diag(S2) # correct
#iS2 = 1 ./ diag(T2) # approx
#iS2 = 1 ./ (T2 - Vt * diag(T1) * Umat) # what clima claims?

for i=1:k
    x_it .= x_it + iS2 .* (rhs - S2*x_it)
    println(norm(x_it-x2_true))
end
x1 = z_b - z_U * x_it

# Also doesnt seem to converge



