module DifferentiableNeuralComputer

import Flux
import Zygote

include("lstm.jl")

struct DNC end

function (dnc::DNC)(x)
    return x
end

end # module
