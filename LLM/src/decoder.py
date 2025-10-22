"""
手写decoder

Created By KennyS
"""

import math
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings(action="ignore")


"""
重点写CausalLM, 所以没有CrossAttention, 省略token embedding这一步

transormer decoder的流程是: input -> self attention -> cross attention -> FFN
caual LM deocer流程是: input -> self attention -> FFN
FFN矩阵有两次变化, 一次升维度, 一次降维度, 其中llama对于GPT的改进, 把GeLU变成了SwishGLU, 多了一个矩阵, 所以升维会从 4h -> 4h * 2 / 3
原版的transformer用post-norm, gpt2 llama用pre-norm, 使用RMSNorm替代LayerNorm
"""

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, num_head, dropout=0.1):
        super().__init__()

        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.dropout = dropout

        self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(self.dropout)

        # FFN
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_layernorm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.act_fn = nn.ReLU()
        self.ffn_dropout = nn.Dropout(self.dropout)

    def attention_output(self, query, key, value, attention_mask=None):
        """
        query: (batch_size, num_head, seq_len, head_dim)
        key: (batch_size, num_head, seq_len, head_dim)
        value: (batch_size, num_head, seq_len, head_dim)
        attention_mask: (batch_size, 1, 1, seq_len)
        """
        # 计算两者相关性
        key = key.transpose(2,3) # [batch_size, num_head, head_dim, seq_len]
        attention_weight = torch.matmul(query, key) / math.sqrt(self.head_dim)

        # attention mask进行调整 变成causal attention
        if attention_mask is not None:
            # 变成下三角矩阵, 因为是causal attention, 所以只需要下三角矩阵
            attention_mask = attention_mask.tril()
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e20"))   
        else:
            # 人工构造一个下三角矩阵
            attention_mask = torch.ones_like(attention_weight).tril()
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float("-1e20"))

        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.attn_dropout(attention_weight)
        
        mid_output = torch.matmul(attention_weight, value) # [batch_size, num_head, seq_len, head_dim]
        mid_output = mid_output.transpose(1,2).contiguous()
        batch_size, seq_len, _, _ = mid_output.size()
        mid_output = mid_output.view(batch_size, seq_len, -1)
        output = self.output_proj(mid_output)
        
        return output

    def attention_block(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        query = self.q_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)
        key = self.k_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)
        value = self.v_proj(x).view(batch_size, seq_len, self.num_head, -1).transpose(1,2)

        output = self.attention_output(
            query, key, value, attention_mask=attention_mask
        )

        return self.layernorm(x + output)
    
    def ffn_block(self, x):
        up = self.act_fn(self.up_proj(x))
        down = self.down_proj(up)
        down = self.ffn_dropout(down)

        return self.ffn_layernorm(x + down)

    def forward(self, x):
        # x输入shape一般都是[batch_size, seq_len, hidden_dim]
        x = self.attention_block(x, attention_mask=None)
        x = self.ffn_block(x)
        return x