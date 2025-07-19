# Tutorial for triton

*Created by KennyS*

---

## 执行结构

### cuda

1. 级别_layer_norm_fwd_fused

- cuda执行级别从grid -> block(blockIdx.x, y, z) -> thread(threadIdx.x, y, z)
- cuda核函数从thread开始，由小到大，计算全局索引：global_idx = blockIdx.x * blockDim.x + threadIdx.x

### triton

1. 级别

- 以program为单位执行
- grid -> program（对应cuda的block tl.program_id(axis)）
- (thread) tl.arange(0, BLOCK_SIZE), 对应block中多少个thread


## layer norm

- 一般输入size: (batch_size, token_num, dim), layer_norm在最后一个维度channel(dim)上做归一化 nn.LayerNorm(dim)
- 在每个样本内部的dim上做归一化，不会在batch中的多个样本之间做归一化

$$
y = \frac {x - E(x)}{\sqrt{Var(x)} + \epsilon} \cdot w + b
$$


## Tensor

### 存储结构

1. 例如一个三行两列的tensor: a = torch.randn(3,2), 在内存中存储为一个平铺结构[a00 a01 a10 a11 a20 a21]. 在第0维度，从一个元素跳到下一个元素，需要经过2个元素，stride=2. 在第1维度，从一个元素跳到下一个元素，需要经过1个元素，stride=1