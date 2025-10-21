"""
手写attention

Created By KennyS
"""

import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


"""
SelfAttention公式:
    Attention(X) = softmax(QK^T / sqrt(d_k)) V
    Q = XWq
    K = XWk
    V = XWv

1. matmul和@是一样的作用
2. 为什么要除以sqrt(d_k)
    - 防止梯度消失
    - 让QK的内积分布保持和输入一样
3. 爱因斯坦方程表达式用法: torch.einsum("...ij,...jk->...ik", Q, K)
4. X.repeat(1,1,3)表示在不同的维度进行repeat操作, 也可以用torch.expand操作
"""


# Version 1. 直接根据公式实现
class SelfAttnV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 一般linear都是默认有bias的
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        # x的final dim可以和hidden_dim不同
        Q = self.query_proj(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_proj(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value_proj(x)  # [batch_size, seq_len, hidden_dim]

        # shape is [batch_size, seq_len, seq_len]
        # torch.matual可以改成 Q @ K.transpose(-1,-2)
        # 其中K需要改成[batch_size, hidden_dim, seq_len]
        attention_value = torch.matmul(Q, K.transpose(-1,-2))
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.hidden_dim), dim=-1
        )
        # shape is [batch_size, seq_len, seq_len]
        output = torch.matmul(attention_weight, V)
        
        return output
        

if __name__ == "__main__":
    attn = SelfAttnV1(4)
    x = torch.randn(3, 2, 4)
    output = attn(x)
    print(output.shape)