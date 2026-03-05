# build an f
function build_rhs(n::Int)
    return randn(n)
end

# Get f from file
function load_rhs_from_serialize(filepath)
    data = deserialize(filepath)
    return data.f
end

"""
    load_rhs_from_serialize(filepath, B1, B2=nothing)

Reconstruct the right-hand side vector `b` from serialized block data.

`B1` and `B2` specify the variable ordering used to assemble the vector.
If `B2 === nothing`, a single vector ordered by `B1` is returned.
If `B2` is provided, the vector is concatenated in the order `[B1; B2]`

Returns the assembled vectors `b1, b2` if `B2` is provided, `b` otherwise.
"""
function load_rhs_from_serialize(filepath, B1, B2=nothing)
    data = deserialize(filepath)

    rhs_arrays = data.rhs_arrays
    sizes = data.sizes
    FT = eltype(data.f)

    rhs_blocks = isnothing(B2) ?
        Dict(:b => B1) :
        Dict(:b1 => B1, :b2 => B2)

    built_rhs = Dict()

    for block in keys(rhs_blocks)
        vars = rhs_blocks[block]
        N = sum(sizes[v] for v in vars)
        b = zeros(FT, N)

        pos = 1
        for v in vars
            n = sizes[v]
            @inbounds b[pos:pos+n-1] .= rhs_arrays[v]
            pos += n
        end

        built_rhs[block] = b
    end

    if !isnothing(B2)
        return built_rhs[:b1], built_rhs[:b2]

    else
        return built_rhs[:b]
    end
end
