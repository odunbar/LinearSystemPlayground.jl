# build an f
function build_rhs(n::Int)
    return randn(n)
end

# Get f from file
function load_rhs_from_serialize(filepath)
    data = deserialize(filepath)
    return data.f
end
