import ClimaCore.MatrixFields

using SparseArrays
using Serialization
using BandedMatrices
using LinearAlgebra

"""
Convert a ClimaCore scalar field name to a simplified string label.

Removes wrappers such as `@name` and `components.data`, replaces some unicode
subscripts, and produces a stable key used in the block and RHS dictionaries.
"""
function scalar_label(name) 
    return replace(
        replace(
            replace(
                string(name),
                "@name" => "",
                "components.data." => "",
                "sgsʲs.:(1)" => "sgs",
                'ₕ' => "h",
                '₃' => "3",
            ),
            ['(', ')', ':'] => "",
        ),
        "u3.1" => "u3",
    )
end

"""
Extract Jacobian blocks and RHS vectors from a Newton integrator state.

Returns a named tuple
    (; block_arrays, rhs_arrays)

- `block_arrays[(row, col)]` contains the Jacobian block for that pair of scalar
  variables (typically `BandedMatrix` or `UniformScaling`).
- `rhs_arrays[var]` contains the RHS vector for that scalar variable.

Variable names are converted to strings using `scalar_label`.
"""
function extract_block_and_rhs_arrays(integrator)
    Y = integrator.u
    t = integrator.t
    (; p) = integrator
    FT = eltype(Y)

    scalar_names = CA.scalar_field_names(Y)

    jacobian_alg = integrator.cache.newtons_method_cache.j.alg

    dtγ = p.dt * FT(CA.LinearAlgebra.diag(integrator.alg.tableau.a_imp)[end])
    column_Y = CA.first_column_view(Y)
    column_p = CA.first_column_view(p)
    
    column_cache = CA.jacobian_cache(jacobian_alg, column_Y, p.atmos; verbose = false)
    CA.update_jacobian!(jacobian_alg, column_cache, column_Y, column_p, dtγ, t)
    column_∂R_∂Y = column_cache.matrix    

    block_arrays = Dict()
    for block_key in Iterators.product(scalar_names, scalar_names)
        block_key in keys(column_∂R_∂Y) || continue
        block_value = Base.materialize(column_∂R_∂Y[block_key])
        block_arrays[(scalar_label(block_key[1]), scalar_label(block_key[2]))] =
            block_value isa Fields.Field ?
            MatrixFields.column_field2array(block_value) : block_value
    end

    column_f = CA.first_column_view(integrator.cache.newtons_method_cache.f)
    rhs_arrays = Dict()
    for rhs_key in scalar_names
        rhs_arrays[scalar_label(rhs_key)] =
            copy(parent(MatrixFields.get_field(column_f, rhs_key)))
    end

    return (; block_arrays, rhs_arrays)
end

"""
Return the list of available variables in a predefined order.

The function filters a fixed ordered list of variables and returns only those
present in `extracted.rhs_arrays`, preserving the predefined order.
"""
function available_variables(extracted)
    ordered_vars = [
        "c.ρ",
        "c.sgs.q_liq",
        "c.sgs.q_ice",
        "c.sgs.q_rai",
        "c.sgs.q_sno",
        "c.sgs.q_tot",
        "c.sgs.mse",
        "c.sgs.ρa",
        "c.ρq_liq",
        "c.ρq_ice",
        "c.ρq_rai",
        "c.ρq_sno",
        "c.ρq_tot",
        "c.ρe_tot",
        "c.ρtke",
        "c.uh.1",
        "c.uh.2",
        "f.sgs.u3",
        "f.u3",
    ]

    rhs_keys = keys(extracted.rhs_arrays)
    @assert Set(rhs_keys) ⊆ Set(ordered_vars) "rhs_keys contains unknown variables"
    return [v for v in ordered_vars if v in rhs_keys]
end

"""
Assemble the global sparse matrix `A` and RHS vector `f` from extracted block
and RHS dictionaries, and save them to a file.

Inputs
- `extracted`: output of `extract_block_and_rhs_arrays`
- `outpath`: file where the assembled system is serialized

Returns `(A, f)` and saves a serialized object containing `A`, `f`,
`offsets`, `sizes`, `block_arrays`, and `rhs_arrays`.
"""
function assemble_and_save_from_dicts(extracted, outpath::AbstractString)
    block_arrays = extracted.block_arrays
    rhs_arrays = extracted.rhs_arrays
    var_order = available_variables(extracted)
    FT = eltype(first(values(rhs_arrays)))

    # --- sizes + offsets ---
    sizes = Dict{String,Int}()
    for v in var_order
        sizes[v] = length(rhs_arrays[v])
    end

    offsets = Dict{String,Int}()
    off = 1
    for v in var_order
        offsets[v] = off
        off += sizes[v]
    end
    N = off - 1

    # --- assemble RHS ---
    f = zeros(FT, N)
    for v in var_order
        r0 = offsets[v]
        rv = rhs_arrays[v]
        @inbounds for i in 1:length(rv)
            f[r0 + i - 1] = FT(rv[i])
        end
    end

    # --- sparse triplets ---
    I = Int[]
    J = Int[]
    V = FT[]

    function add_uniform_scaling!(α, row0, col0, nrow, ncol)
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

    function add_banded!(B::BandedMatrix, row0, col0)
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

    # --- assemble matrix ---
    for vi in var_order
        row0 = offsets[vi]
        nrow = sizes[vi]

        for vj in var_order
            col0 = offsets[vj]
            ncol = sizes[vj]

            key = (vi, vj)
            haskey(block_arrays, key) || continue
            blk = block_arrays[key]

            if blk isa UniformScaling
                add_uniform_scaling!(blk.λ, row0, col0, nrow, ncol)

            elseif blk isa BandedMatrix
                add_banded!(blk, row0, col0)

            else
                error("Unsupported block type $(typeof(blk)) for key ($vi,$vj)")
            end
        end
    end

    A = sparse(I, J, V, N, N)

    # --- save EVERYTHING ---
    data = (
        A = A,
        f = f,
        var_order = var_order,
        offsets = offsets,
        sizes = sizes,
        block_arrays = block_arrays,
        rhs_arrays = rhs_arrays,
    )

    open(outpath, "w") do io
        serialize(io, data)
    end

    return A, f
end

extracted_blocks_and_rhs_dicts = extract_block_and_rhs_arrays(integrator)
A, f = assemble_and_save_from_dicts(extracted_blocks_and_rhs_dicts, "linear_system_rico_1M.jls")

# How to load the data from file:
# data = deserialize(open("newton_system.jls"))
# A = data.A
# f = data.f
# block_arrays = data.block_arrays
# rhs_arrays = data.rhs_arrays