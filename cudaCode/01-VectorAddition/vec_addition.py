import torch
import triton
import triton.language as tl

@triton.jit # 类似于cuda的__global__
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) # 获取当前Program ID 对应cuda的block索引
    # 计算当前block处理的起始索引
    block_start = pid * BLOCK_SIZE
    # 生成当前block内的元素偏移量 对应cuda的thread索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 生成mask 避免处理超过向量长度的元素
    mask = offsets < n_elements
    # 从GPU显存加载a和b的对应元素 自动处理边界
    a_vals = tl.load(a + offsets, mask=mask)
    b_vals = tl.load(b + offsets, mask=mask)
    # 执行逐元素相加
    c_vals = a_vals + b_vals
    # 将结果写回GPU显存
    tl.store(c + offsets, mask=mask, val=c_vals)

   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),) # triton向上取整除法 确保所有元素都被覆盖
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE) # [grid]指定Block数量 ()传递参数
