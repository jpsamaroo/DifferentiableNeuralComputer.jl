using Flux: softmax

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

mutable struct Access
    MemMat       # R^{N*W}
    LinkMat      # R^{N*N}
    readWts      # RH element Array of [0, 1]^N
    wrtWt        # WH element Array of (0, 1)^{N}
    usageVec     # (0, 1)^{N}
    precedenceWt # (0, 1)^{N}
    readVecs
    numWriteHeads
    numReadHeads
    wordSize
    memorySize
end

function Access(memorySize=128, wordSize=20, numReadHeads=1, numWriteHeads=1)
    MemMat = zeros(Float32, memorySize, wordSize)
    LinkMat = zeros(Float32, memorySize, memorySize)

    readWts = rand(Float32, memorySize, numReadHeads)
    fill!(readWts, 1f-6)

    wrtWt = rand(Float32, memorySize, numWriteHeads)
    fill!(wrtWt, 1f-6)

    usageVec = zeros(Float32, memorySize, numWriteHeads)
    fill!(usageVec, 1f-6)
    precedenceWt = zeros(Float32, memorySize)

    readVecs = zeros(Float32, wordSize, numReadHeads)
    fill!(readVecs, 1f-6)

    Access(MemMat, LinkMat, readWts, wrtWt, usageVec, precedenceWt, readVecs,
            numWriteHeads, numReadHeads, wordSize, memorySize)
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
                    readHeads,                    # free gates
                    writeHeads,                   # allocation gates
                    writeHeads,                   # write gates
                    readHeads * (1 + 2writeHeads) # read modes
                    ])


    readkeys       =         interfaceVec[demarcations[1]+1:demarcations[2]]
    readstrengths  = oneplus(interfaceVec[demarcations[2]+1:demarcations[3]])
    writekeys      =         interfaceVec[demarcations[3]+1:demarcations[4]]
    writestrengths = oneplus(interfaceVec[demarcations[4]+1:demarcations[5]])
    eraseVec       =       σ.(interfaceVec[demarcations[5]+1:demarcations[6]])
    writeVec       =         interfaceVec[demarcations[6]+1:demarcations[7]]
    freeGts        =       σ.(interfaceVec[demarcations[7]+1:demarcations[8]])
    allocGt        =       σ.(interfaceVec[demarcations[8]+1:demarcations[9]])
    writeGt        =       σ.(interfaceVec[demarcations[9]+1:demarcations[10]])
    readmodes      = softmax(interfaceVec[demarcations[10]+1:demarcations[11]])

    readkeys  = reshape(readkeys, wordSize, readHeads)   # W * RH
    writekeys = reshape(writekeys, wordSize, writeHeads) # W * WH
    eraseVec  = reshape(eraseVec, wordSize, writeHeads)  # W * WH
    writeVec  = reshape(writeVec, wordSize, writeHeads)  # W * WH
    readmodes = reshape(readmodes, (1+2writeHeads), readHeads)  # (WH for backward + WH for forward + 1 for content lookup) * RH

    return (readkeys=readkeys, readstrengths=readstrengths, writekeys=writekeys,
            writestrengths=writestrengths, eraseVec=eraseVec, writeVec=writeVec,
            freeGts=freeGts, allocGts=allocGt, writeGts=writeGt, readmodes=readmodes)
end

function (access::Access)(interfaceVec)
    # dynamic memory allocation

    interface = interfacedisect(interfaceVec, access.numWriteHeads, access.wordSize, access.numReadHeads)

    memRetVec = prod(1 .- interface[:freeGts]' .* access.readWts, dims=2)  # Memory Retention Vector = [0, 1]^{N}
    access.usageVec = (access.usageVec .+ access.wrtWt .- access.usageVec .* access.wrtWt) .* memRetVec
    freelist = sortperm(access.usageVec)  # Z^{N}
    allocWt = zeros(access.usageVec)
    @. allocWt[freelist] = (1 - access.usageVec[freelist]) * cumprod([1; access.usageVec[freelist]][1:end-1])  # (0, 1)^{N}

    # writing
    wrtcntWt = memprobdistrib(access.MemMat, interface[:writekeys], interface[:writestrengths]) # Write content weighting = (0, 1)^{N}
    access.wrtWt .= interface[:writeGts] * (interface[:allocGts] * allocWt + (1 - interface[:allocGts])*wrtcntWt)
    @. access.MemMat *= (ones(access.MemMat) - access.wrtWt*interface[:eraseVec]') #  First we erase...
    @. access.MemMat += access.wrtWt*interface[:writeVec]' # Then we write.

    # temporal linkage
    eye = Matrix{Float32}(I, size(access.LinkMat)...)
    prevlinkscale = @. 1 - access.wrtWt - access.wrtWt'
    newlink = @. access.wrtWt * access.precedenceWt'
    @. access.LinkMat = (1 - eye) * (prevlinkscale * access.LinkMat + newlink)

    access.precedenceWt = (1 - sum(access.wrtWt)) .* access.precedenceWt .+ access.wrtWt

    # reading
    forwardWts = [access.LinkMat * readWt for readWt in access.readWts]
    backwardWts = [access.LinkMat' * readWt for readWt in access.readWts]
    readcntWts = memprobdistrib.([access.MemMat], readkeys, readstrengths) # Read content weightings

    access.readWts = [π[1].*b .+ π[2].*readcntWts .+ π[3].*f for (π, b, f) in (interface[:readmodes], backwardWts, forwardWts)]
    access.readvecs = [access.MemMat' * W_r for W_r in access.readWts]

    return(readvecs)
end
