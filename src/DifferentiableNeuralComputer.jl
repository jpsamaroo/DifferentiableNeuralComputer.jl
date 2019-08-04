module DifferentiableNeuralComputer

import Flux
import Zygote

struct DNC end

function (dnc::DNC)(x)
    return x
end

end # module
