import torch
import triton
import triton.language as tl


"""
Demo for triton.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1. vector add
@triton.jit # 相当于global
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    block_size: tl.constexpr
):
    """
    核函数

    Args:
        x_ptr: 输入张量的指针，指向张量中第一个元素的起始位置
        y_ptr: 输入张量的指针
        otuput_ptr: 输出张量的指针
        n_elements: 向量中的元素总数
        block_size: 每个block处理的元素数量
    """
    pid = tl.program_id(axis=0) # program_id 相当于BlockIdx.x
    block_start = pid * block_size # 当前block处理数据  的起始索引
    offsets = block_start + tl.arange(0, block_size) # 相当于全局索引 blockIdx.x*blockDim.x+threadIdx.x, 内存偏移量
    mask = offsets < n_elements # 防止越界, 最后一个block可能不满
    x = tl.load(x_ptr + offsets, mask=mask) # mask确保安全访问
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(
        x: torch.Tensor,
        y: torch.Tensor
):
    """
    Tensor加法
    """
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel() # 计算总元素数量
    grid = lambda meta: (triton.cdiv(n_elements, meta['block_size']), ) # 决定kernel启动多少个block

    add_kernel[grid](x, y, output, n_elements, block_size=1024)

    return output

# --------------------------------------------------------------------------------#

# 2. fused-softmax
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE:tl.constexpr,
    num_stages:tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        
# --------------------------------------------------------------------------------#

# 3. layer_norm
@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    # 获取 blockIdx.x
    row = tl.program_id(0) # x方向上program的id (行)
    X += row * stride # stride=8192
    Y += row * stride # 参考内存中的平铺结构

    # 计算mean
    mean = 0
    _mean =tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    
    # 计算rstd
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    _var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(vars + eps)

    # write
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


def layer_norm(x, weight, bias, eps):
    y = torch.empty_like(x) # (1151, 8192)
    x_arg = x.reshape(-1, x.shape[-1]) # 1151 8192 没有起到什么作用
    M, N = x_arg.shape # 1151 8192
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device) # (1151)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device) # (1151)

    MAX_FUSED_SIZE = 65536 // x.element_size() # 65536 // 2(float16) 
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N)) # 2的多少次方是大于N的 -> 2^{13}=8192
    # BLOCK_SIZE不能超过MAX_FUSED_SIZE
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB...")
    
    # kernel func
    # x_arg.stride(0)=8192 x_arg.stride(1)=1
    _layer_norm_fwd_fused[(M, )](x_arg, y, weight, bias, mean, rstd, x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE)

    return y
     
if __name__ == "__main__":

    # 1. vector add
    # torch.manual_seed(0)
    # size = 98432
    # x = torch.rand(size, device=DEVICE)
    # y = torch.rand(size, device=DEVICE)
    # output_torch = x + y
    # output_triton = add(x, y)
    # print(output_torch)
    # print(output_triton)
    # print(f'The maximum difference between torch and triton is '
    #     f'{torch.max(torch.abs(output_torch - output_triton))}')

# --------------------------------------------------------------------------------#

    # 2. 

# --------------------------------------------------------------------------------#

    # 3. layer norm
    M = 1151
    N = 8192
    dtype = torch.float16
    eps = 1e-5
    device = "cuda"

    x_shape = (M, N) # (batch_size*token_num, dim)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)

    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    y_tri = layer_norm(x, weight, bias, eps)
    print(y_tri)
