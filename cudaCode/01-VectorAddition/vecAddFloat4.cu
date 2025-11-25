#include <cuda_runtime.h>
#include <stdio.h>

// 启用向量化合并访存。每个thread处理4个元素，因此blockPerGrid变少

#define BLOCK_SIZE 1024 // 每个线程块处理的元素数量
#define CEIL(a, b) ((a + b - 1) / b)
#define CONST_FLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])  // 定义 FLOAT4 宏

// A和B是const float*类型 表示指向只读浮点数据的指针
// FLOAT4宏中使用reinterpret_cast<float4*>(&(value)) 试图将const float的地址转换为float4*(非const) 违反了const的限定符规则 不能通过转换移除指针的const属性
// 解决: 需要将转换后的指针也声明为const 使用const float4* 调整相关读取方式


__global__ void vecAddKernel(const float *A, const float *B, float *C, int N)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // 全局线程索引
    if (idx < N) // 边界检查 防止数组越界
    {
        float4 tmp_A = CONST_FLOAT4(A[idx]);
        float4 tmp_B = CONST_FLOAT4(B[idx]);
        // 计算flaot4的逐元素相加
        float4 tmp_C;
        tmp_C.x = tmp_A.x + tmp_B.x;
        tmp_C.y = tmp_A.y + tmp_B.y;
        tmp_C.z = tmp_A.z + tmp_B.z;
        tmp_C.w = tmp_A.w + tmp_B.w;
        // 写回结果
        reinterpret_cast<float4*>(&C[idx])[0] = tmp_C;
    }
}


void vecAdd(float *A, float *B, float *C, int N){
  dim3 threadPerBlock = BLOCK_SIZE;
  dim3 blockPerGrid = CEIL(CEIL(N,4), BLOCK_SIZE);	// modify

  vecAddKernel<<<blockPerGrid, threadPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}


// 简化的测试函数（测试 N=8，确保对齐）
void simple_test() {
    const int N = 8;
    float h_A[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_B[N] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float h_C_gpu[N] = {0};
    float h_C_cpu[N] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    vecAdd(d_A, d_B, d_C, N);

    cudaMemcpy(h_C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }

    printf("Input A: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_A[i]);
    printf("\nInput B: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_B[i]);
    printf("\nGPU Result: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_C_gpu[i]);
    printf("\nCPU Result: ");
    for (int i = 0; i < N; i++) printf("%.1f ", h_C_cpu[i]);

    bool pass = true;
    const float eps = 1e-5;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > eps) {
            pass = false;
            break;
        }
    }
    printf("\n\nTest: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    simple_test();
    return 0;
}