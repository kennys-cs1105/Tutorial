#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>


#define BLOCK_SIZE 32 // 每个线程块处理的元素数量
#define CEIL(a, b) ((a + b - 1) / b)

// A[M*K], B[K*N], C[M*N]

__global__ void matMulKernel(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i < K; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    } 
}

void matMul(float *A, float *B, float *C, int M, int N, int K)
{
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockPerGrid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    matMulKernel<<<blockPerGrid, threadPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}


// 测试函数 验证M=2 N=3 K=4
void test_matMul()
{
    const int M = 2, N = 3, K = 4;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;

    // 初始化CPU数据
    float h_A[size_A];
    for (int i = 0; i < size_A; i++){
        h_A[i] = i + 1.0f;
    }

    float h_B[size_B];
    for (int i = 0; i < size_B; i++){
        h_B[i] = i * 0.1f;
    }

    float h_C_gpu[size_C] = {0};
    float h_C_cpu[size_C] = {0};

    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // 拷贝数据到GPU
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // 调用核函数
    matMul(d_A, d_B, d_C, M, N, K);

    // 拷贝结果回GPU
    cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算CPU结果
    for (int row = 0; row < M; row++){
        for (int col = 0; col < N; col++){
            float sum = 0.0f;
            for (int i = 0; i < K; i++){
                sum += h_A[row * K + i] * h_B[i * N + col];
            }
            h_C_cpu[row * N + col] = sum;
        }
    }

    // 打印输出
    printf("Matrix A (%dx%d):\n", M, K);
    for (int i = 0; i < M; i++){
        for (int j = 0; j < K; j++) printf("%5.1f", h_A[i * K + j]);
        printf("\n");
    }

    printf("\nMatrix B (%dx%d):\n", K, N);
    for (int i = 0; i < K; i++){
        for (int j = 0; j < N; j++) printf("%5.1f", h_B[i * N + j]);
        printf("\n");
    }

    printf("\nGPU Result (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++) printf("%5.1f", h_C_gpu[i * N + j]);
        printf("\n");
    }

    // 验证结果
    bool pass = true;
    const float eps = 1e-5;
    for (int i = 0; i < size_C; i++){
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > eps){
            pass = false;
            break;
        }
    }
    printf("\nTest %s\n", pass ? "PASS" : "FAIL");

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    test_matMul();
    return 0;
}