import math
from torch import nn
import torch

import ms_blkq4linear_ext


class BlkQ4Linear(nn.Module):
    def __init__(self, block_size, col_wise, scale_only, weights, bias_tensor=None):
        super(BlkQ4Linear, self).__init__()
        print(ms_blkq4linear_ext.__file__)
        self.block_size = block_size
        self.col_wise = col_wise
        self.has_offsets = not scale_only
        self.in_features = weights.size(1)
        self.out_features = weights.size(0)

        self.q_weights, self.q_scales, self.q_zp = ms_blkq4linear_ext.quant(block_size, col_wise, self.has_offsets, weights)
        self.bias = bias_tensor

    def forward(self, input):
        output = ms_blkq4linear_ext.forward(self.block_size, self.col_wise, self.in_features, self.out_features, input, self.q_weights, self.q_scales, self.q_zp)
        if self.bias is not None:
            output += self.bias
        return output
