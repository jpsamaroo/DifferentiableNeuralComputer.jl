# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

module DifferentiableNeuralComputer

include("access.jl")

@kwdef struct DNCState
    access_output
    access_state
    controller_state
end

struct DNC
    _controller
    _access
    _access_output_size
    _clip_value::Real
    _output_size
    _state_size::DNCState
end
function DNC(access_config, controller_config, output_size, clip_value=0f0)
    dnc = new()
    dnc._controller = DNCLSTM(controller_config)
    dnc._access = MemoryAccess(access_config)

    dnc._access_output_size = prod(dnc._access.output_size)
    dnc._output_size = output_size
    @assert clip_value >= 0 "clip_value cannot be negative"
    dnc._clip_value = clip_value

    dnc._output_size = tf.TensorShape([output_size])
    dnc._state_size = DNCState(
        access_output=dnc._access_output_size,
        access_state=dnc._access.state_size,
        controller_state=dnc._controller.state_size)

    return dnc
end

clip_if_enabled(dnc::DNC, x) =
    dnc._clip_value > 0 ?
        clamp(x, -dnc._clip_value, dnc._clip_value) : x

"""
Args:
  inputs: Tensor input.
  prev_state: A `DNCState` tuple containing the fields `access_output`,
      `access_state` and `controller_state`. `access_state` is a 3-D Tensor
      of shape `[batch_size, num_reads, word_size]` containing read words.
      `access_state` is a tuple of the access module's state, and
      `controller_state` is a tuple of controller module's state.

Returns:
  A tuple `(output, next_state)` where `output` is a tensor and `next_state`
  is a `DNCState` tuple containing the fields `access_output`,
  `access_state`, and `controller_state`.
"""
# FIXME: Fold `prev_state` into `DNC`
function (dnc::DNC)(inputs, prev_state::DNCState)
    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    batch_flatten = snt.BatchFlatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output, controller_state = dnc._controller(
        controller_input, prev_controller_state)

    controller_output = clip_if_enabled(dnc, controller_output)
    controller_state = tf.contrib.framework.nest.map_structure(dnc._clip_if_enabled, controller_state)

    access_output, access_state = dnc._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    output = snt.Linear(
        output_size=first(dnc._output_size),
        name="output_linear")(output)
    output = clip_if_enabled(dnc, output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state)
end

function initial_state(dnc::DNC, batch_size, dtype=Float32)
    return DNCState(
        access_output=zeros(dtype, dnc._access.output_size .+ batch_size),
        access_state=initial_state(dnc._access, batch_size, dtype),
        controller_state=initial_state(dnc._controller, batch_size, dtype)
    )
end

state_size(dnc::DNC) = dnc._state_size
output_size(dnc::DNC) = dnc._output_size


end # module
