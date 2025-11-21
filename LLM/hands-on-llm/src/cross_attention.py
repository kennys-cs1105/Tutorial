"""
交叉注意力 Cross Attention
广泛应用于:
1. Transformer编码器-解码器结构
2. 图像与文本对齐, 例如Stable Diffusion、CLIP等  
3. 跨模态信息融合
4. 序列间交互建模

与自注意力不同的是, 交叉注意力的作用是让序列A从序列B中提取信息

Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
其中Q来自序列A, K,V来自序列B

工作流程:
假设有:
    图像特征: x_img -> [batch, img_len, dim]
    文本特征: x_txt -> [batch, txt_len, dim]

计算方式:
    Query = Wq * x_img
    Key = Wk * x_txt
    Value = Wv * x_txt
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    out = Attention(Q,K,V)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x_query, x_context):
        """
        x_query: [batch, query_len, dim] Query端序列, 例如图像
        x_context: [batch, context_len, dim] Key Value端序列, 例如文本
        """
        B, Q_len, dim = x_query.shape
        _, C_len, _ = x_context.shape

        # 线性变换得到Q,K,V
        q = self.q_proj(x_query)  # [B, Q_len, dim]
        k = self.k_proj(x_context)  # [B, C_len, dim]
        v = self.v_proj(x_context)  # [B, C_len, dim]

        # 分头
        def _split_head(x):
            return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = _split_head(q)
        k = _split_head(k)
        v = _split_head(v)

        # 计算注意力分数
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Q_len, dim)

        # 线性变换输出
        out = self.out_proj(out)

        return out



if __name__ == "__main__":
    # 测试模型
    model = CrossAttention(dim=256, num_heads=8)
    x_query = torch.randn(1, 32, 256)
    x_context = torch.randn(1, 20, 256)
    out = model(x_query, x_context)
    print(out.shape)  
