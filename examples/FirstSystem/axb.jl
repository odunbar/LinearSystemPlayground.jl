# build/load matrix pkg
using LinearAlgebra
using LinearMaps
using SparseArrays
using Random
using Serialization
using BandedMatrices

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
#=
A, T1, U, Vt, T2 = build_a(n)
b = build_rhs(size(A,2))
=#

# load from file
filepath = "../../matrix_vector_pairs/linear_system_trmm_0M.jls"
B1, B2 = ["c.ρ", "c.sgs.q_tot", "c.sgs.mse", "c.sgs.ρa", "c.ρq_tot", "c.ρe_tot", "c.ρtke", ], ["c.uh.1", "c.uh.2", "f.u3", "f.sgs.u3"]

#filepath = "../../matrix_vector_pairs/linear_system_trmm_1M.jls"
# filepath = "../../matrix_vector_pairs/linear_system_rico_1M.jls"
A = load_a_from_serialize(filepath)
# b = load_rhs_from_serialize(filepath)
T1, U, Vt, T2 = load_a_from_serialize(filepath, B1, B2)
b = load_rhs_from_serialize(filepath, B1, B2)
# -----------
# Full solves: A
# ----------- 

# [1] Naive solve
@info "Naive solve: A\\b"
@btime $A \ $b
@allocated A \ b
x_true = A\b

# [2] Iterative solve - no precond
@info "iterative solve (no precond): gmres(A,b)"
@btime gmres($A, $b)
@allocated gmres(A,b)
x = gmres(A,b)
@info "error $(norm(x - x_true))"

# [3] Iterative solve - with basic left precond
@info "iterative solve (ilu precond): gmres(A,b, Pl=P)"
@btime begin
    P = ilu($A)              # ILU(0)
    gmres($A, $b, Pl=P)  # left preconditioning
end
P = ilu(A)              # ILU(0)
@allocated gmres(A,b, Pl=P)
x = gmres(A, b, Pl=P)
@info "error $(norm(x - x_true))"      

# [4] Jacobi solve
@info "Fixed-point solve (diag(A) precond)"
k=20
Ad = diag(A)
iAd = 1 ./ Ad
x_it = zeros(size(A,2))

M = I - iAd .* A
spect = maximum(abs, eigvals(Matrix(M)))
@info "spectral radius: $(maximum(abs, eigvals(Matrix(M)))) must be <1 to converge)"
if spect < 1
    for i=1:k
        x_it .= x_it + iAd.*(b - A*x_it)
        println(norm(x_it-x))
    end
    x = x_it
    @info "error $(norm(x - x_true))"
else
    @warn "skipping test due to spectral radius $(spect) > 1"
end
# I wont time as this iteration blows up

# ------------
# Schur solves: S & T1
# ------------

# split rhs
s1 = size(T1,1) 
b1, b2 = b[1:s1], b[s1+1:end]
Umat = Matrix(U)


# [1] Naive solve
@info "Naive solve, Schur: T1\\b1, T1\\U, S2 \\ rhs"
function do_schur_solve(T1, U, Vt, T2, Umat, b1, b2)
    F1 = factorize(T1)
    z_b = F1 \ b1
    z_U = F1 \ Umat
    S2 = T2 - Vt * z_U 
    x2 = S2 \ (b2 - Vt*z_b)
    x1 = z_b - z_U * x2
    return [x1;x2]
end

@btime begin
    do_schur_solve($T1, $U, $Vt, $T2, $Umat, $b1, $b2)
end
x = do_schur_solve(T1, U, Vt, T2, Umat, b1, b2)
@info "err: $(norm(x_true - x))"

# [2] GMRES with (Schur with T2 precond), and allocates
@info "GMRES (T2 precond) on Schur"
function do_gmres_schur_solve_PT2(T1, U, Vt, T2, Umat, b1, b2)
    F1 = lu(T1)
    schur_action(x) = T2 * x - Vt * (F1 \ (U*x))
    schur_adj_action(x) = T2' * x - U' * (F1 \ (Vt'*x))
    
    S2 = LinearMap(
        schur_action,
        schur_adj_action,
        size(T2,1),
        size(T2,1),
    )

    PT2 = ilu(T2)
    rhs = b2 - Vt*(F1\b1)
    x2 = gmres(S2, rhs ; Pl=PT2)
    x1 = F1 \ (b1-Umat*x2)
    return [x1;x2]
end

@info "GMRES (S2 precond) on Schur"
@btime begin
    do_gmres_schur_solve_PT2($T1, $U, $Vt, $T2, $Umat, $b1, $b2)
end

x=do_gmres_schur_solve_PT2(T1, U, Vt, T2, Umat, b1, b2)
@info "err: $(norm(x_true - x)))"

# [3] Jacobi
#true soln
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
iS2 = Diagonal(1 ./ diag(S2)) # correct
#iS2 = Diagonal(1 ./ diag(T2)) # approx
#iS2 = (Matrix(T2 - Vt * (iTd .* U))) # what clima claims?

# check spectral radii
M = I - iS2*S2
spect = maximum(abs, eigvals(M))
@info "spectral radius: $(maximum(abs, eigvals(Matrix(M)))) must be <1 to converge)"
if spect < 1
    for i=1:k
        x_it .= x_it + iS2 * (rhs - S2*x_it)
        println(norm(x_it-x2_true))
    end
    x1 = z_b - z_U * x_it
else
    @warn "skipping test due to spectral radius $(spect) > 1"
end
# Also doesnt seem to converge



