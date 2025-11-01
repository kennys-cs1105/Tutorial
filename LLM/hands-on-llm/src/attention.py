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
        

# Version 2. 优化效率
# QKV矩阵计算时, 可以合并成一个大矩阵进行运算
class SelfAttnV2(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # 一般linear都是默认有bias的
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [batch, seq, hidden_dim]
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attention_weight = torch.softmax(
            Q @ K.transpose(-1,-2) / math.sqrt(self.hidden_dim), dim=-1
        )
        output = attention_weight @ V

        return self.output_proj(output)
    

# Version 3. 加入细节
# attention计算时有dropout, 比较奇怪的位置
# attention计算时一般会加入attention_mask, 因为一些样本会进行padding
# MHSA过程中, 除了QKV, 还有一个output对应的投影矩阵
class SelfAttnV3(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # 一般linear都是默认有bias的
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        # x: [batch, seq, hidden_dim]
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.hidden_dim, dim=-1)
        attention_weight = torch.softmax(
            Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim), dim=-1
        )

        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e20"))
        
        attention_weight = torch.softmax(attention_mask, dim=-1)
        attention_weight = self.dropout(attention_weight)
        output = attention_weight @ V

        return self.output_proj(output)


# Version 4. Multi Head Self Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_num, num_head) -> None:
        super().__init__()

        self.num_head = num_head
        self.head_dim = hidden_num // num_head
        self.hidden_num = hidden_num

        self.q_proj = nn.Linear(hidden_num, hidden_num)
        self.k_proj = nn.Linear(hidden_num, hidden_num)
        self.v_proj = nn.Linear(hidden_num, hidden_num)
        self.output_proj = nn.Linear(hidden_num, hidden_num)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq, hidden_dim]
        batch_size, seq_len, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # shape变成 [batch_size, num_head, seq_len, head_dim]
        q_state = Q.view(batch_size, seq_len, self.num_head, self.head_dim).permute(0,2,1,3)
        k_state = K.view(batch_size, seq_len, self.num_head, self.head_dim).permute(1,2)
        v_state = V.view(batch_size, seq_len, self.num_head, self.head_dim).permute(1,2)

        attention_weight = torch.softmax(
            q_state @ k_state.transpose(-1,-2) / math.sqrt(self.head_dim)
        )
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e20"))
        
        attention_weight = torch.softmax(attention_weight, dim=3)
        attention_weight = self.dropout(attention_weight)

        output_mid  =attention_weight @ v_state
        # 重新变成[batch_size, seq_len, num_head, head_dim]
        # contiguous返回一个连续内存的tensor, view只能在连续内存中操作
        output_mid = output_mid.transpose(1,2).contiguous()
        # 变成[batch_size, seq_len, hidden_num]
        output = output_mid.view(batch_size, seq_len, -1)
        output = self.output_proj(output)

        return output



if __name__ == "__main__":
    attn = SelfAttnV1(4)
    x = torch.randn(3, 2, 4)
    output = attn(x)
    print(output.shape)