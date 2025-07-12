import torch
import triton
import triton.language as tl


"""
Demo for triton.
"""


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
    """
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(x_ptr + offsets, mask=mask)
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
    
     
    