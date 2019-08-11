module DifferentiableNeuralComputer

using Flux, Zygote, LinearAlgebra

include("DNCLSTM.jl")

cosine_sim(u, v) = (u'v)/(norm(u)*norm(v))

# content-based addressing
function C(M, k, β)
    out = [cosine_sim(k, M[i, :]) for i in 1:size(M, 1)] .* β
    out = softmax(out)
end

struct DNC
    M
    L
    readweights
    W_w
    u
    p
end

function (dnc::DNC)(readkeys, readstrengths, k_w, β_w, erase, write, freegates, g_a, g_w, readmodes)
    # dynamic memory allocation
    ψ = prod(1 .- freegates .* dnc.readweights)
    dnc.u = (dnc.u .+ dnc.W_w .- dnc.u .* dnc.W_w) .* ψ
    Ø = sortperm(dnc.u)
    a = zeros(dnc.u)
    a[Ø] .= (1 .- dnc.u[Ø]) .* cumprod([1; dnc.u[Ø]][1:end-1])

    # writing
    c_w = C(dnc.M, k_w, β_w)
    dnc.W_w = g_w * (g_a .* a .+ (1 - g_a)*c_w)
    dnc.M = dnc.M .* (ones(dnc.M) - dnc.W_w*erase') .+ dnc.W_w*write'

    # temporal linkage
    for (i, j) in Tuple.(CartesianIndices(dnc.L))
        if i == j dnc.L[i, j] = 0 # exclude self links
        else dnc.L[i, j] = (1 - dnc.W_w[i] - dnc.W_w[j])*dnc.L[i, j] + dnc.W_w[i] * p[j] end
    end
    dnc.p = (1 - sum(dnc.W_w)) .* dnc.p .+ dnc.W_w

    # reading
    f = [dnc.L * W_r for W_r in dnc.readweights]
    b = [dnc.L' * W_r for W_r in dnc.readweights]
    c_r = C.([dnc.M], readkeys, readstrengths)

    dnc.readweights = [π[1].*b .+ π[2].*c .+ π[3].*f for (π, b, f) in (readmodes, b, f)]
    readvecs = [dnc.M' * W_r for W_r in dnc.readweights]
    
    return(readvecs)
end

end # module
