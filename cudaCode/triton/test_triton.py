import torch
import triton
import triton.language as tl


"""
Demo for triton.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1. vector add
@triton.jit
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
        x_ptr: 输入张量的指针
        y_ptr: 输入张量的指针
        otuput_ptr: 输出张量的指针
        n_elements: 元素总数
        block_size: 每个block处理的元素数量
    """
    pid = tl.program_id(axis=0) # 当前block的id
    block_start = pid * block_size # 当前block处理的起始索引
    offsets = block_start + tl.arange(0, block_size) # 当前block负责的所有索引
    mask = offsets < n_elements # 防止越界, 最后一个block可能不满
    x = tl.load(x_ptr + offsets, mask=mask)
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
        



     
if __name__ == "__main__":

    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
