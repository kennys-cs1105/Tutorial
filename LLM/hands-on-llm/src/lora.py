"""
手写lora

Created By KennyS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
无论是LLM还是diffusionm, 都需要对模型的参数进行微调, 而lora就是一种参数高效的微调方法.
lora可以节约显存, 训练快, 效果损失较小
    1. 减少显存的主要原因是训练参数变小了, 比如只对qkv层做lora

核心原理: 
    1. 对原始模型的参数进行分解, 得到低秩矩阵
    2. 训练低秩矩阵, 得到微调参数
    3. 合并低秩矩阵, 得到微调后的模型参数

W_new =W_0 + A @ B
h = W_0 @ x + A @ B @ x
h = (W_0 + a/rAB) @ x
"""


class LinearLoRALayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            merge: bool = False,
            rank: int = 8,
            alpha: float = 16,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.alpha = alpha
        self.rank = rank

        # linear weight的shape是 [output_features, input_features], 正确做法是 x W^T
        self.linear = nn.Linear(in_features, out_features)

        if rank > 0:
            # lora_a 和 lora_b 是可训练的参数
            self.lora_a = nn.Parameter(
                torch.zeros(out_features, rank)
            )
            # lora_a需要初始化为高斯分布
            nn.init.kaiming_normal(self.lora_a, a=0.01)

            self.lora_b = nn.Parameter(
                torch.zeros(rank, in_features)
            )
            self.scale = self.alpha / self.rank

            # linear需要设置为不可训练
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False
        
        self.dropout = nn.Dropout(
            dropout
        ) if dropout > 0 else nn.Identity()

        # 如果采用merge进行推理
        # 那么会把lora_a 和 lora_b 两个小矩阵的参数直接放到linear.weight中
        if merge: 
            self.merge_weight()
        
    def forward(self, x):
        # x shape [batch, seq_len, in_features]
        if self.rank > 0 and not self.merge:
            output = self.linear(x) + self.scale * (x @ (self.lora_a @ self.lora_b).T)
        elif self.rank > 0 and self.merge:
            output = self.linear(x)
        else:
            output = self.linear(x)
        
        return self.dropout(output)

    def merge_weight(self):
        # 合并lora_a 和 lora_b 两个小矩阵的参数, 放到linear.weight中
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)
    
    def unmerge_weight(self):
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)


