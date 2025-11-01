"""
手写GQA

Created By KennyS
"""

import math
import torch
import torch.nn as nn


"""
手写大模型组件GQA(Group Query Attention)
效果损失小, 推理的时候可以加速(kvcache小)
当num_key_value_head = 1 时, 就是mqa
当num_head = num_key_value_head 时, 就是mha
"""


class GroupQueryAttention(nn.Module):
    def __init__(self, num_head, num_key_value_head, hidden_dim):
        super().__init__()
        
        assert hidden_dim % num_head == 0, "hidden_dim must be divisible by num_head" # 可以整除
        assert num_head % num_key_value_head == 0, "num_head must be divisible by num_key_value_head" # N个query head为一组
        
        self.num_head = num_head
        self.num_key_value_head = num_key_value_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_head

        self.q_proj = nn.Linear(hidden_dim, self.num_head * self.head_dim)
        # k v out shape (num_key_value_head * head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.num_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.num_key_value_head * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        # x shape [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.size()

        # qkv projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # attention_weight 目标shape [batch_size, num_head, seq_len, seq_len]
        q = q.view(batch_size, seq_len, self.num_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_head, self.head_dim)

        q = q.transpose(1, 2) # [batch_size, num_head, seq_len, head_dim]
        k = k.transpose(1, 2) # [batch_size, num_key_value_head, seq_len, head_dim]
        v = v.transpose(1, 2) # [batch_size, num_key_value_head, seq_len, head_dim]

        k = k.repeat_interleave(self.num_head // self.num_key_value_head, dim=1) # [batch_size, num_head, seq_len, head_dim]
        v = v.repeat_interleave(self.num_head // self.num_key_value_head, dim=1) # [batch_size, num_head, seq_len, head_dim]

        attention_weight = (q @ k.transposeF(2,3)) / math.sqrt(self.head_dim)
        attention_weight = torch.softmax(attention_weight, dim=-1)

        output = attention_weight @ v
        output = self.o_proj(output.view(batch_size, seq_len, -1))
        
        return output

