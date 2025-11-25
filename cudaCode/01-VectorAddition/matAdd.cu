#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

// 二维矩阵元素相加
// 需要熟悉row/col计算index

#define BLOCK_SIZE 32 // 每个线程块处理的元素数量
#define CEIL(a, b) ((a + b - 1) / b)

__global__ void vecAddKernel(const float* A, const float* B, float* C, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 全局线程索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程索引
    if (row < M && col < N) // 边界检查 防止数组越界
    {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx]; // 逐元素相加
    }
}


void matAdd(float *A, float *B, float *C, int M, int N) {
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);  // 修正为 dim3 构造函数
    dim3 blockPerGrid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    vecAddKernel<<<blockPerGrid, threadPerBlock>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}

// 简化的测试函数 仅测试N=5
// 测试函数（验证 M=3, N=5 的小矩阵）
void test_MatAdd() {
    const int M = 3;
    const int N = 5;
    const int total = M * N;

    // 初始化主机数据
    float h_A[total], h_B[total], h_C_gpu[total], h_C_cpu[total];
    for (int i = 0; i < total; i++) {
        h_A[i] = i + 1.0f;       // A = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
        h_B[i] = i * 0.1f;       // B = [[0.0,0.1,0.2,0.3,0.4], [0.5,0.6,...], ...]
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, total * sizeof(float));
    cudaMalloc(&d_B, total * sizeof(float));
    cudaMalloc(&d_C, total * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, total * sizeof(float), cudaMemcpyHostToDevice);

    // 执行 GPU 计算
    matAdd(d_A, d_B, d_C, M, N);

    // 拷贝结果回主机
    cudaMemcpy(h_C_gpu, d_C, total * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 计算参考结果
    for (int i = 0; i < total; i++) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    // 打印结果 (仅打印前两行示例)
    printf("Matrix A (First 2 rows):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < N; j++) printf("%5.1f", h_A[i * N + j]);
        printf("\n");
    }
    printf("\nMatrix B (First 2 rows):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < N; j++) printf("%5.1f", h_B[i * N + j]);
        printf("\n");
    }
    printf("\nGPU Result (First 2 rows):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < N; j++) printf("%5.1f", h_C_gpu[i * N + j]);
        printf("\n");
    }

    // 验证结果
    bool pass = true;
    const float eps = 1e-5;
    for (int i = 0; i < total; i++) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > eps) {
            pass = false;
            printf("Error at index %d: GPU=%.2f, CPU=%.2f\n", i, h_C_gpu[i], h_C_cpu[i]);
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
    test_MatAdd();
    return 0;
}