using Flux
using Flux: glorot_uniform, @treelike, Recur

mutable struct DNCLstmCell
    Wi; Wf; Ws; Wo
    bi; bf; bs; bo
    h; s # These aren't used in the forward pass but need to be included for Recur.
end
@treelike DNCLstmCell

function DNCLstmCell(in::Integer, hidden::Integer; init=glorot_uniform)
    DNCLstmCell([init(hidden, in+2*hidden) for i in 1:4]...,
                [zeros(hidden) for i in 1:6]...)
end

#=
    h: hidden at previous timestep
    s: state at previous timestep
    h′: state at previous layer (at same timestep)
    x: original input to the LSTM
=#
function(c::DNCLstmCell)((h, s), h′, x)
    h = size(h)==size(h′) ? h : repeat(h, 1, size(x, 2)) # support for batching.

    i = σ.(c.Wi * [x; h; h′] .+ c.bi)
    f = σ.(c.Wf * [x; h; h′] .+ c.bf)
    s′ = f .* c.s .+ i .* tanh.(c.Ws * [x; h; h′] .+ c.bs)
    o = σ.(c.Wo * [x; h; h′] .+ c.bo)
    h′ = o .* tanh.(s′)
    return((h′, s′), (h′, x))
end

Flux.hidden(m::DNCLstmCell) = (m.h, m.s)
DNCLstm(a...; ka...) = Recur(DNCLstmCell(a...; ka...))

#=
Note: for creating a multi-layered DNCLSTM, the arguments of each layer
    need to be compatible. In other words, the hidden state has to keep
    the same dimensions and the input size stays constant since x gets
    passed through each layer.
    e.g.: Chain(DNCLSTM(16,10), DNCLSTM(16,10))
=#
