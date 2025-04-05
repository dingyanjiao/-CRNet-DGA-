import torch
import torch.nn as nn

class LCFFN(nn.Module):
    def __init__(self, d_model, expansion_factor=4, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.delight = nn.GELU()
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        
        # 深度可分离卷积（Depthwise + Pointwise）
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model * expansion_factor
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=d_model * expansion_factor,
            out_channels=d_model * expansion_factor,
            kernel_size=1  
        )
        
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)

    def forward(self, x):
        x = self.norm(x)
        x = self.delight(x)
        x = self.linear1(x)  # (batch, seq_len, d_model * expansion)
        
        # 调整维度以适配 Conv1d
        x = x.transpose(1, 2)  # (batch, d_model * expansion, seq_len)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model * expansion)
        
        x = self.linear2(x)
        return x