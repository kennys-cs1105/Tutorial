#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 1024 // 每个线程块处理的元素数量
#define CEIL(a, b) ((a + b - 1) / b)

__global__ void vecAddKernel(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    if (idx < N) // 边界检查 防止数组越界
    {
        C[idx] = A[idx] + B[idx]; // 逐元素相加
    }
}


void vecAdd(const float* A, const float* B, float* C, int N)
{
    dim3 threadPerBlock(BLOCK_SIZE);
    dim3 blockPerGrid(N, BLOCK_SIZE);
    vecAddKernel<<<blockPerGrid, threadPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

// 简化的测试函数 仅测试N=5
void simple_test()
{
    const int N = 5;

    // 固定输入
    float h_A[N] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_B[N] = {0.1, 0.2, 0.3, 0.4, 0.5};

    float h_C_gpu[N] = {0};
    float h_C_cpu[N] = {0};

    // GPU计算
    float *d_A, *d_B, *d_C;
    
    // 分配内存
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // 复制数据到GPU
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核函数
    vecAdd(d_A, d_B, d_C, N);

    cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU计算
    for (int i = 0; i < N; i++)
    {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // 输出结果
    printf("Input A: ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", h_A[i]);
    }
    printf("\n");
    printf("Input B: ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", h_B[i]);
    }
    printf("\n");
    printf("Output C (GPU): ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", h_C_gpu[i]);
    }
    printf("\n");
    printf("Output C (CPU): ");
    for (int i = 0; i < N; i++)
    {
        printf("%f ", h_C_cpu[i]);
    }
    printf("\n");

    // 验证
    bool pass = true;
    const float eps = 1e-5;
    for (int i = 0; i < N; i++){
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > eps){
            pass = false;
            break;
        }
    }
    printf("\nTest: %s\n", pass ? "PASS" : "FAIL");

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    simple_test();
    return 0;
}