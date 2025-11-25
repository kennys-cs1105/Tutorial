#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    if (idx < N) // 边界检查 防止数组越界
    {
        C[idx] = A[idx] + B[idx]; // 逐元素相加
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int N)
{
    int threadsPerBlock = 256; // 每个线程块中的线程数 32线程为一个warp 256线程为8个warp
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 计算需要的线程块数

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N); // 核函数启动语法 <>内是网格/块的维度配置 ()内是传递给核函数的参数
    cudaDeviceSynchronize(); // 同步操作 阻塞CPU 等待GPU上的核函数完成 避免CPU提前访问未计算完成的结果
}