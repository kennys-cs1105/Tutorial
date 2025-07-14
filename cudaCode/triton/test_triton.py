import torch
import triton
import triton.language as tl


"""
Demo for triton.
1. vector add
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
