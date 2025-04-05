# Copyright (c) 2022 Microsoft
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)
import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
# 函数返回正弦和余弦函数的值作为位置嵌入的一部分

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2] # 将每两个相邻的元素分别放在不同的张量中
    x = torch.stack((-x2, x1), dim=-1) # 将每两个相邻元素进行旋转
    if x.shape[-1]%2 == 1:
        # fill last dim with zero if hidden_size is odd
        x2 = torch.concat((x2, torch.zeros_like(x2[:, :, :1])), dim=-1) # 处理 hidden_size 为奇数的情况。如果 hidden_size 为奇数，那么在旋转后会存在一个多余的元素，第一行代码检查是否为奇数，第二行代码在 x2 的最后一维上添加了一个零元素，以保证维度匹配
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\ 
# 在原始张量的每两个相邻元素之间进行了旋转

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m 
# 将输入矩阵的每个元素进行复制，然后将复制后的元素交替插入到原始矩阵的相邻位置

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos)) # 对输入的正弦和余弦部分分别进行缩放和复制插入操作
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos[:, :x.shape[-1]]) + (rotate_every_two(x) * sin)[:, :, :x.shape[-1]] # 对于输入 x，将其与缩放后的余弦部分 cos 相乘，并与经过 rotate_every_two 操作后的 x 与缩放后的正弦部分 sin 相乘后的结果相加
# 这段代码的目的是在输入张量上应用旋转的位置嵌入

class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x
    
# test
if __name__ == "__main__":
    x = torch.eye(4).unsqueeze(0)
    xpos = XPOS(4)
    x_rot = xpos(x)
    # apply reverse
    x_rot_rev = xpos.forward(x)

    print(x_rot @ x_rot_rev.transpose(-1, -2))
# 根据输入张量x及其长度计算位置编码。然后，使用计算出的正弦和余弦嵌入将旋转位置嵌入应用于输入张量，并可选择缩小位置编码。返回修改后的张量。