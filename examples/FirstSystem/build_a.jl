
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


# Get A from file
function load_a_from_serialize(filepath)
    data = deserialize(filepath)
    A = data.A
    
    return data.A
end

function add_uniform_scaling!(α, row0, col0, nrow, ncol, I, J, V)
    FT=eltype(V)
    n = min(nrow, ncol)
    @inbounds for i in 1:n
        a = FT(α)
        if a != 0.0
            push!(I, row0 + i - 1)
            push!(J, col0 + i - 1)
            push!(V, a)
        end
    end
end

function add_banded!(B::BandedMatrix, row0, col0, I, J, V)
    FT=eltype(V)
    nrow, ncol = size(B)
    kl, ku = bandwidths(B)
    @inbounds for j in 1:ncol
        i1 = max(1, j - ku)
        i2 = min(nrow, j + kl)
        for i in i1:i2
            a = FT(B[i, j])
            if a != 0.0
                push!(I, row0 + i - 1)
                push!(J, col0 + j - 1)
                push!(V, a)
            end
        end
    end
end

function build_local_offsets(vars, sizes)
    offsets = Dict{String,Int}()
    pos = 1
    for v in vars
        offsets[v] = pos
        pos += sizes[v]
    end
    return offsets
end

"""
    build_a_from_serialize(filepath, B1, B2=nothing)

Reconstruct sparse matrix blocks from serialized block data.

`B1` and `B2` specify the variable ordering used to assemble the matrix.
If `B2 === nothing`, a single matrix `A` with ordering `B1` is returned
(e.g. a block-permuted version of the original matrix).

If `B2` is provided, the matrix is assembled in block form

    [ T1  U
      Vt  T2 ]

where `T1`, `U`, `Vt`, and `T2` correspond to interactions between
variables in `B1` and `B2`.

Returns `(T1, U, Vt, T2)` if `B2` is given, otherwise `A`.
"""
function load_a_from_serialize(filepath, B1, B2=nothing)
    data = deserialize(filepath)

    # rebuild matrix blocks from keys
    var_order = data.var_order
    if !isnothing(B2)
        block_var_orders = Dict(
            :T1=>[B1, B1], #T1
            :U=>[B1, B2], #U
            :Vt=>[B2, B1], #Vt
            :T2=>[B2, B2], #T2
        )
    else
        block_var_orders = Dict(
            :A=>[B1, B1],
        )
    end
    offsets = data.offsets
    sizes = data.sizes
    block_arrays = data.block_arrays
    built_blocks=Dict()
    FT = eltype(data.f)
    for block in keys(block_var_orders)
        I = Int[]
        J = Int[]
        V = FT[]
        N1 = sum(sizes[bs] for bs in block_var_orders[block][1])
        N2 = sum(sizes[bs] for bs in block_var_orders[block][2])
        local_row_offsets = build_local_offsets(block_var_orders[block][1], sizes)
        local_col_offsets = build_local_offsets(block_var_orders[block][2], sizes)
        
        for vi in block_var_orders[block][1]
            row0 = local_row_offsets[vi]
            nrow = sizes[vi]
            
            for vj in block_var_orders[block][2]
                col0 = local_col_offsets[vj]
                ncol = sizes[vj]
                
                key = (vi, vj)
                haskey(block_arrays, key) || continue
                blk = block_arrays[key]
                
                if blk isa UniformScaling
                    add_uniform_scaling!(blk.λ, row0, col0, nrow, ncol, I, J, V)
                    
                elseif blk isa BandedMatrix
                    add_banded!(blk, row0, col0, I, J, V)
                    
                else
                    error("Unsupported block type $(typeof(blk)) for key ($vi,$vj)")
                end
            end
        end

        #to get block-local indices, take off the block start index
        built_blocks[block] = sparse(I, J, V, N1, N2)
        
    end

    if !isnothing(B2)
        return built_blocks[:T1], built_blocks[:U], built_blocks[:Vt], built_blocks[:T2]
    else
        return built_blocks[:A]
    end
end
