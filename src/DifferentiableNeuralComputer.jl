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

oneplus(x::AbstractVecOrMat) = log.(exp.(x) .+ 1)

struct DNC
    MemMat       # R^{N*W}
    LinkMat      # R^{N*N}
    readWts      # Array of R^{W}
    wrtWt        # (0, 1)^{N}
    usageVec     # (0, 1)^{N}
    precedenceWt # (0, 1)^{N}
    numWriteHeads
    numReadHeads
    wordSize
end

"""
    interfacedisect(interfaceVec, writeHead, wordSize, readHeads)

Disects the interface vector obtained as the output to obtain various memory
access controls for the DNC.
"""
function interfacedisect(interfaceVec, writeHeads, wordSize, readHeads)
    demarcations = cumsum([0,                     # Starting Index
                    readHeads*wordSize,           # read keys
                    readHeads,                    # read strengths
                    writeHeads*wordSize,          # write keys
                    writeHeads,                   # write strengths
                    writeHeads*wordSize,          # erase vectors
                    writeHeads*wordSize,          # write vectors
                    readHeaads,                   # free gates
                    writeHeads,                   # allocation gates
                    writeHeads,                   # write gates
                    readHeads * (1 + 2writeHeads) # read modes
                    ])


    readkeys       =         interfaceVec[demarcations[1]+1:demarcations[2]]
    readstrengths  = oneplus(interfaceVec[demarcations[2]+1:demarcations[3]])
    writekeys      =         interfaceVec[demarcations[3]+1:demarcations[4]]
    writestrengths = oneplus(interfaceVec[demarcations[4]+1:demarcations[5]])
    eraseVec       =       σ(interfaceVec[demarcations[5]+1:demarcations[6]])
    writeVec       =         interfaceVec[demarcations[6]+1:demarcations[7]]
    freeGts        =       σ(interfaceVec[demarcations[7]+1:demarcations[8]])
    allocGt        =       σ(interfaceVec[demarcations[8]+1:demarcations[9]])
    writeGt        =       σ(interfaceVec[demarcations[9]+1:demarcations[10]])
    readmodes      = softmax(interfaceVec[demarcations[10]+1:demarcations[11]])

    readkeys  = reshape(readkeys, readHeads, wordSize)   # RH * W
    writekeys = reshape(writekeys, writeHeads, wordSize) # WH * W
    eraseVec  = reshape(eraseVec, writeHeads, wordSize)  # WH * W
    writeVec  = reshape(writeVec, writeHeads, wordSize)  # WH * W
    readmodes = reshape(readmodes, readHeads, (1+2writeHeads))  # RH * (WH for backward + WH for forward + 1 for content lookup)

    return (readkeys=readkeys, readstrengths=readstrengths, writekeys=writekeys,
            writestrengths=writestrengths, eraseVec=eraseVec, writeVec=writeVec,
            freeGts=freeGts, allocGts=allocGts, writeGts=writeGts, readmodes=readmodes)
end

function (dnc::DNC)(input)
    # dynamic memory allocation

    interface = interfacedisect(input, dnc.numWriteHeads, dnc.wordSize, dnc.numReadHeads)

    memRetVec = prod(1 .- interface[:freeGts] .* dnc.readWts)  # Memory Retention Vector = [0, 1]^{N}
    dnc.usageVec = (dnc.usageVec .+ dnc.wrtWt .- dnc.usageVec .* dnc.wrtWt) .* memRetVec
    freelist = sortperm(dnc.usageVec)  # Z^{N}
    allocWt = zeros(dnc.usageVec)
    @. allocWt[freelist] = (1 - dnc.usageVec[freelist]) * cumprod([1; dnc.usageVec[freelist]][1:end-1])  # (0, 1)^{N}

    # writing
    wrtcntWt = memprobdistrib(dnc.MemMat, interface[:writekeys], interface[:writestrengths]) # Write content weighting = (0, 1)^{N}
    dnc.wrtWt .= interface[:writeGts] * (interface[:allocGts] * allocWt + (1 - interface[:allocGts])*wrtcntWt)
    @. dnc.MemMat *= (ones(dnc.MemMat) - dnc.wrtWt*interface[:eraseVec]') #  First we erase...
    @. dnc.MemMat += dnc.wrtWt*interface[:writeVec]' # Then we write.

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

    dnc.readWts = [π[1].*b .+ π[2].*readcntWts .+ π[3].*f for (π, b, f) in (interface[:readmodes], backwardWts, forwardWts)]
    readvecs = [dnc.MemMat' * W_r for W_r in dnc.readWts]

    return(readvecs)
end

end # module
