#include("DNCLSTM.jl")
include("Access.jl")

using Distributions: TruncatedNormal
using Flux

mutable struct DNC
    controller
    access::Access
    interfaceVec
end

function DNC(memory_size=16, word_size=16, num_read_heads=4, num_write_heads=1,
                hidden_size=64, output_size=4, input_size=4)
    controller_input_size = input_size + word_size * num_read_heads

    # A truncated normal distribution with no elements further than 2σ of μ.
    dist = TruncatedNormal(0, 0.01, -0.02, 0.02)

    # The interface vector of this dimension will only work when the number of
    # write heads is equal to 1. I have yet to figure out how this value changes
    # as the number of write heads change.
    interface_vec_dimensions = word_size * num_read_heads + 3word_size + 5num_read_heads + 3
    controller_output_size = output_size + interface_vec_dimensions
    controller = LSTM(controller_input_size, controller_output_size)
    access = Access(memory_size, word_size, num_read_heads, num_write_heads)

    interface_vec = rand(dist, interface_vec_dimensions)
    DNC(controller, access, interface_vec)
end

function (dnc::DNC)(input)
    readVecs = dnc.access.readVecs

    # flattening readVecs
    readVecs = reshape(readVecs, readVecs |> size |> prod)

    # concatinating them with input to form controller input
    controller_input = [input; readVecs]

    controller_output = dnc.controller(controller_input)


end
