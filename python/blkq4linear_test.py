#
# Copyright (c) Microsoft Corporation. All rights reserved.
#

import torch
import torch.nn as nn

import ms_blkq4linear

in_features = 64
out_features = 32

# Create a linear layer with 28*28 input features and 20 output features
layer1 = nn.Linear(in_features=in_features, out_features=out_features, bias=False, dtype=torch.float16)

weights = layer1.weight.clone().detach()

# numbers from a quantization block come from a single row
# but keep in mind that weights are stored in column major, 
col_wise = False
block_size = 32
scale_only = False

block_rows = block_size if col_wise else 1
block_cols = 1 if col_wise else block_size

q_w_rows = in_features // 2
q_w_cols = out_features
q_weights = torch.empty((q_w_cols, q_w_rows), dtype=torch.uint8)

q_m_rows = in_features // block_rows
q_m_cols = out_features // block_cols
q_scales = torch.empty((q_m_cols, q_m_rows), dtype=torch.float16)
q_zp = torch.empty((q_m_cols, q_m_rows), dtype=torch.uint8)

def next_i4(val: int) -> int:
    # making the cycle 13 instead of 16, avoiding same values in a row
    val = (val + 5) % 16
    if val == 11 or val == 7 or val == 3:
        val = (val + 5) % 16
    return val

v = 7
for c in range(q_w_cols):
    for r in range(q_w_rows):
        v0 = v
        v = next_i4(v)
        v1 = v
        v = next_i4(v)
        q_weights[c][r] = (v1 << 4) | v0

for c in range (q_m_cols):
    for r in range(q_m_rows):
        f = (((c * v + r + v // 3) % 63) + 1)
        v += 41
        m = c * v + r + v * 3
        q_scales[c][r] = f / (1 << (4 + (m % 2)))
        if scale_only:
            q_zp[c][r] = 8
        else:
            q_zp[c][r] = ((f + m + v) % 8) + 4

for c in range(out_features):
    for r in range(in_features):
        w_r = r // 2
        scale_c = c // block_cols
        scale_r = r // block_rows
        offset = int(q_zp[scale_c][scale_r])
        w = 0
        if r % 2 == 0:
            w = int(q_weights[c][w_r] & 0x0f)
        else:
            w = int(q_weights[c][w_r] >> 4)
        scale = float(q_scales[scale_c][scale_r])
        weights[c][r] = scale * float(w - offset)
        # print(f'({r},{c}) w={w}, offset={offset}, scale={scale}, {weights[c][r]}')


weights = weights.to(torch.device('cuda:0'))

blkq4l_module = ms_blkq4linear.BlkQ4Linear(block_size, col_wise, scale_only, weights)

torch.set_printoptions(profile="full")
print(blkq4l_module.q_weights)
torch.set_printoptions(profile="default")


