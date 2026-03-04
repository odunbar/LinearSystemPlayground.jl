
# Diagonally dominant tridiagonal matrix
function tridiag_dd(n)
    dl = rand(n-1)
    du = rand(n-1)
    d  = rand(n) .+ abs.([0; dl]) .+ abs.([du; 0]) .+ 1.0
    return spdiagm(-1 => -dl, 0 => d, 1 => -du)
end

# Diagonally dominant bidiagonal matrix
function bidiag_dd(n)
    dl = rand(n-1)
    d  = rand(n) .+ abs.([0; dl]) .+ 1.0
    return spdiagm(-1 => -dl, 0 => d)
end


function build_a(n::Int)
    Z = spzeros(n,n)
    
    I_block = sparse(I,n,n)
    
    T1_22 = tridiag_dd(n)
    T1_23 = tridiag_dd(n)
    T1_32 = Z
    T1_33 = tridiag_dd(n)

    # T1 block (3x3 blocks)
    T1 = [
        I_block   Z         Z;
        Z         T1_22     T1_23;
        Z         Z         T1_33
    ]
    
    # U block (3x2 blocks)
    U = [
        tridiag_dd(n)   bidiag_dd(n);
        tridiag_dd(n)   bidiag_dd(n);
        tridiag_dd(n)   bidiag_dd(n)
    ]
    
    # Vᵀ block (2x3 blocks)
    Vt = [
    Z   Z   Z;
        bidiag_dd(n)  bidiag_dd(n)  bidiag_dd(n)
    ]
    
    # T2 block (2x2 blocks)
    T2 = [
        tridiag_dd(n)   Z;
        bidiag_dd(n)    tridiag_dd(n)
    ]
    
    # ----------------------------
    # Full 5x5 block matrix
    # ----------------------------
    
    A = [
        T1  U;
        Vt  T2
    ]
    
    println("Matrix size: ", size(A))
    println("Number of nonzeros: ", nnz(A))
    println("Condition number ", cond(Matrix(A)))
    return A, T1, U, Vt, T2
end
