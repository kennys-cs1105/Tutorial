import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    """
    使用pytorch实现GPU上向量的按元素加法比较直接 直接进行A+B即可
    pytorch会自动对两个GPU Tensor进行按元素加法, 由于A和B已经在GPU上, 因此这个操作会直接在GPU上进行
    C.copy_(A+B) 会将A和B的按元素加法结果复制到C中
    在提供的函数中 N参数表示A B C的长度(元素数量), 但在当前函数中并未显式使用, 这是因为Tensor自带形状信息, 可通过shape访问, 框架会自动根据Tensor的shape进行元素级计算
    但是N参数的存在更多的是为了明确函数的输入约束, 在pytorch代码中起到文档化的作用, 在底层cuda实现中可用于控制内存分配
    """
    assert A.numel() == N and B.numel() == N and C.numel() == N, "向量长度与N不匹配"

    assert A.device.type == 'cuda', "A不在GPU上"
    assert B.device.type == 'cuda', "B不在GPU上"
    assert C.device.type == 'cuda', "C不在GPU上"

    return C.copy_(A+B)


if __name__ == "__main__":
    N = 1000
    A = torch.rand(N, device="cuda")
    B = torch.rand(N, device="cuda")
    C = torch.zeros(N, device="cuda")
    solve(A, B, C, N)
    print(C)