module DifferentiableNeuralComputer

using Flux, Zygote, LinearAlgebra

include("DNCLSTM.jl")

cosine_sim(u, v) = (u'v)/(norm(u)*norm(v))

# content-based addressing
"""
    memprobdistrib(MemMat, key, keystrength)

Defines a normalized probability distribution over the memory locations.
"""
function memprobdistrib(M, k, β)
    out = [cosine_sim(k, M[i, :]) for i in 1:size(M, 1)] .* β
    out = softmax(out)
end

struct DNC
    MemMat       # R^{N*W}
    LinkMat      # R^{N*N}
    readWts      # Array of R^{W}
    wrtWt        # (0, 1)^{N}
    usageVec     # (0, 1)^{N}
    precedenceWt # (0, 1)^{N}
end

function (dnc::DNC)(readkeys, readstrengths, wrtkey, wrtstrength, eraseVec, wrtVec, freeGt, allocGt, wrtGt, readmodes)
    # dynamic memory allocation

    memRetVec = prod(1 .- freeGt .* dnc.readWts)  # Memory Retention Vector = [0, 1]^{N}
    dnc.usageVec = (dnc.usageVec .+ dnc.wrtWt .- dnc.usageVec .* dnc.wrtWt) .* memRetVec
    freelist = sortperm(dnc.usageVec)  # Z^{N}
    allocWt = zeros(dnc.usageVec)
    @. allocWt[freelist] = (1 - dnc.usageVec[freelist]) * cumprod([1; dnc.usageVec[freelist]][1:end-1])  # (0, 1)^{N}

    # writing
    wrtcntWt = memprobdistrib(dnc.MemMat, wrtkey, wrtstrength) # Write content weighting = (0, 1)^{N}
    dnc.wrtWt .= wrtGt * (allocGt * allocWt + (1 - allocGt)*wrtcntWt)
    @. dnc.MemMat *= (ones(dnc.MemMat) - dnc.wrtWt*eraseVec') #  First we erase...
    @. dnc.MemMat += dnc.wrtWt*wrtVec' # Then we write.

    # temporal linkage
    eye = Matrix{Float32}(I, size(dnc.LinkMat)...)
    prevlinkscale = @. 1 - dnc.wrtWt - dnc.wrtWt'
    newlink = @. dnc.wrtWt * dnc.precedenceWt'
    @. dnc.LinkMat = (1 - eye) * (prevlinkscale * dnc.LinkMat + newlink)

    dnc.precedenceWt = (1 - sum(dnc.wrtWt)) .* dnc.precedenceWt .+ dnc.wrtWt

    # reading
    forwardWts = [dnc.LinkMat * readWt for readWt in dnc.readWts]
    backwardWts = [dnc.LinkMat' * readWt for readWt in dnc.readWts]
    readcntWts = memprobdistrib.([dnc.MemMat], readkeys, readstrengths) # Read content weightings

    dnc.readWts = [π[1].*b .+ π[2].*readcntWts .+ π[3].*f for (π, b, f) in (readmodes, backwardWts, forwardWts)]
    readvecs = [dnc.MemMat' * W_r for W_r in dnc.readWts]

    return(readvecs)
end

end # module
