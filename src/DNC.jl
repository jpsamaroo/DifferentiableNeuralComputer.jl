#include("DNCLSTM.jl")
include("Access.jl")

using Distributions: TruncatedNormal
using Flux

mutable struct DNC
    controller # Find and update its grad
    access::Access
    interfaceVecWts#::Array{Float32, T} where T # Find and update its grad
    outputWts#::Array{Float32, } # Find and update its grad
    readVecWts # Find and update its grad
end

function DNC(memory_size=16, word_size=16, num_read_heads=4, num_write_heads=1,
                hidden_size=64, output_size=4, input_size=4)
    controller_input_size = input_size + word_size * num_read_heads

    # A truncated normal distribution with no elements further than 2σ of μ.
    dist = TruncatedNormal(0, 0.1, -0.2, 0.2)

    # The interface vector of this dimension will only work when the number of
    # write heads is equal to 1. I have yet to figure out how this value changes
    # as the number of write heads change.
    interface_vec_dimensions = word_size * num_read_heads + 3word_size + 5num_read_heads + 3
    controller_output_size = output_size + interface_vec_dimensions
    controller = LSTM(controller_input_size, controller_output_size)
    access = Access(memory_size, word_size, num_read_heads)

    interfaceVecWts = param(Float32.(rand(dist, interface_vec_dimensions, controller_output_size)))
    outputWts = param(Float32.(rand(dist, output_size, controller_output_size)))
    readVecWts = param(Float32.(rand(dist, output_size, num_read_heads*word_size)))
    DNC(controller, access, interfaceVecWts, outputWts, readVecWts)
end

function (dnc::DNC)(input::AbstractArray)
    readVecs = dnc.access.readVecs

    # flattening readVecs
    readVecs = reshape(readVecs, length(readVecs))

    # concatinating them with input to form controller input
    controller_input = [input; readVecs]

    controller_output = dnc.controller(controller_input)

    interfaceVec = dnc.interfaceVecWts * controller_output
    output = dnc.outputWts * controller_output

    readvecs = dnc.access(interfaceVec)
    readvecs = reshape(readvecs, length(readvecs))
    readvecs = dnc.readVecWts * readvecs

    output + readvecs
end
