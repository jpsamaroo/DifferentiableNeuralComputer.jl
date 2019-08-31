using Flux: softmax
using LinearAlgebra: norm

cosine_sim(u, v) = (u'v)/(norm(u)*norm(v))

# content-based addressing
"""
    memprobdistrib(MemMat, key, keystrength)

Defines a normalized probability distribution over the memory locations.
"""
function memprobdistrib(M, k, β)
    out = [cosine_sim(k, M[:, i])[1] for i in 1:size(M, 2)] .* β
    out = softmax(out)
end

oneplus(x::AbstractVecOrMat) = log.(exp.(x) .+ 1)

mutable struct Access
    MemMat       # R^{N*W}
    LinkMat      # R^{N*N}

    readWts      # RH element Array of [0, 1]^N    # Find and update its grads
    wrtWt        # WH element Array of (0, 1)^{N}  # Find and update its grads


    usageVec     # (0, 1)^{N}
    precedenceWt # (0, 1)^{N}
    readVecs
    #numWriteHeads
    numReadHeads
    wordSize
    memorySize
end

function Access(memorySize=128, wordSize=20, numReadHeads=1)
    MemMat = zeros(Float32, wordSize, memorySize)
    fill!(MemMat, 1f-6)

    LinkMat = zeros(Float32, memorySize, memorySize)

    readWts = rand(Float32, memorySize, numReadHeads)
    fill!(readWts, 1f-6)

    wrtWt = rand(Float32, memorySize)
    fill!(wrtWt, 1f-6)

    usageVec = zeros(Float32, memorySize)
    fill!(usageVec, 1f-6)

    precedenceWt = zeros(Float32, memorySize)

    readVecs = zeros(Float32, wordSize, numReadHeads)
    fill!(readVecs, 1f-6)

    Access(MemMat, LinkMat, readWts, wrtWt, usageVec, precedenceWt, readVecs,
            numReadHeads, wordSize, memorySize)
end

"""
    interfacedisect(interfaceVec, writeHead, wordSize, readHeads)

Disects the interface vector obtained as the output to obtain various memory
access controls for the DNC.
"""
function interfacedisect(interfaceVec, wordSize=16, readHeads=4)
    demarcations = cumsum([0,                     # Starting Index
                    readHeads*wordSize,           # read keys
                    readHeads,                    # read strengths
                    wordSize,                     # write keys
                    1,                   # write strengths
                    wordSize,          # erase vectors
                    wordSize,          # write vectors
                    readHeads,                    # free gates
                    1,                   # allocation gates
                    1,                   # write gates
                    readHeads * (1 + 2) # read modes
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

    readkeys  = reshape(readkeys, wordSize, readHeads)    # RH * W
    #writekeys = reshape(writekeys, wordSize, writeHeads) # W * WH
    #eraseVec  = reshape(eraseVec, wordSize, writeHeads)  # W * WH
    #writeVec  = reshape(writeVec, wordSize, writeHeads)  # W * WH
    readmodes = reshape(readmodes, readHeads, (1+2))  # (WH for backward + WH for forward + 1 for content lookup) * RH


    return (readkeys=readkeys, readstrengths=readstrengths, writekeys=writekeys,
            writestrengths=writestrengths, eraseVec=eraseVec, writeVec=writeVec,
            freeGts=freeGts, allocGts=allocGt, writeGts=writeGt, readmodes=readmodes)
end

function (access::Access)(interfaceVec::AbstractArray)
    # dynamic memory allocation

    interface = interfacedisect(interfaceVec, access.wordSize, access.numReadHeads)

    memRetVec = prod(1 .- interface[:freeGts]' .* access.readWts, dims=2)  # Memory Retention Vector = [0, 1]^{N}
    access.usageVec = (access.usageVec .+ access.wrtWt .- access.usageVec .* access.wrtWt) .* memRetVec


    ## Generalize it for the multiple write heads
    allocWt = zero(access.usageVec)
    freelist = sortperm(access.usageVec[:, 1])  # Z^{N}
    allocWt[freelist] = (1 .- access.usageVec[freelist]) .* cumprod([1; access.usageVec[freelist]][1:end-1])  # (0, 1)^{N}

    # writing
    wrtcntWt = memprobdistrib(access.MemMat, interface[:writekeys], interface[:writestrengths]) # Write content weighting = (0, 1)^{N}
    access.wrtWt = @. interface[:writeGts] * (allocWt * interface[:allocGts] + (1 - interface[:allocGts])' * wrtcntWt)
    access.MemMat .*= ones(Float32, size(access.MemMat)) - access.wrtWt*interface[:eraseVec]' #  First we erase...
    access.MemMat += access.wrtWt*interface[:writeVec]' # Then we write.

    # temporal linkage
    eye = one(access.LinkMat)
    prevlinkscale = @. 1 - access.wrtWt - access.wrtWt' ## Have to do something here for multiple write heads...
    newlink = @. access.wrtWt * access.precedenceWt'
    @. access.LinkMat = (1 - eye) * (prevlinkscale * access.LinkMat + newlink)

    access.precedenceWt = (1 - sum(access.wrtWt)) .* access.precedenceWt .+ access.wrtWt

    # reading
    forwardWts = access.LinkMat * access.readWts
    backwardWts = access.LinkMat' * access.readWts

    ## make this pretty
    readcntWts = [memprobdistrib(access.MemMat, interface[:readkeys][:, i], interface[:readstrengths][i]) for i=1:access.numReadHeads] # Read content weightings
    readcntWts = hcat(readcntWts...)

    ## Make this pretty...
    access.readWts = [interface[:readmodes][i, 1].*backwardWts[:, i] .+ interface[:readmodes][i, 2].*readcntWts[:, i] .+ interface[:readmodes][i, 3].*forwardWts[:, i] for i=1:access.numReadHeads]
    access.readWts = hcat(access.readWts...)
    access.readVecs = access.MemMat * access.readWts
    return(access.readVecs)
end
