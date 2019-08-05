module DifferentiableNeuralComputer

import Flux
import Zygote

include("src/lstm.jl")

struct DNC end

function (dnc::DNC)(x)
    return x
end

end # module
