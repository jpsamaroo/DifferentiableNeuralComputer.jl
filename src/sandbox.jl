# Prepping training data -> random memory pattern
#include("DNCLSTM.jl")

in = 4
#model = DNCLSTM(in, 4)

num_seq = 10
seq_len = 6
seq_width = 4
con = rand(1:seq_width, seq_len)

seq = zeros(seq_len, seq_width)
idx = con |> enumerate .|> CartesianIndex
seq[idx] .= 1
zer = zeros(seq_len, seq_width)

final_i_data = hcat(seq', zer')
final_o_data = hcat(zer', seq')




###################
#@time allocWt * interface[:allocGts] + wrtcntWt * (1 .- interface[:allocGts])'
#@time interface[:allocGts]' .* allocWt + (1 .- interface[:allocGts])' .* wrtcntWt
