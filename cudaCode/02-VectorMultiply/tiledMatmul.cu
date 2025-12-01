#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>


#define BLOCK_SIZE 32 // 每个线程块处理的元素数量
#define CEIL(a, b) ((a + b - 1) / b)


__global__ void matMulTiledKernel(float *A, float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int numTiles = CEIL(K, BLOCK_SIZE);

    for (int t = 0; t < numTiles; t++){
        int offset = t * BLOCK_SIZE;

        // 加载分块A
        int loadA_col = offset + threadIdx.x;
        if (row < M && loadA_col < K){
            As[threadIdx.y][threadIdx.x] = A[row * K + loadA_col];
        } else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载分块B
        int loadB_row = offset + threadIdx.y;
        if (col < N && loadB_row < K){
            Bs[threadIdx.y][threadIdx.x] = B[loadB_row * N + col];
        } else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算分块结果
        for (int k = 0; k < BLOCK_SIZE; k++){
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 存储结果
    if (row < M && col < N){
        C[row * N + col] = sum;
    }
}


void matMul(float *A, float *B, float *C, int M, int N, int K)
{
    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blockPerGrid(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    matMulTiledKernel<<<blockPerGrid, threadPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}


// 测试函数 M=2 N=3 K=4
void test_MatMul()
{
    const int M = 2;
    const int N = 3;
    const int K = 4;
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

    // CPU计算结果
    for (int row = 0; row < M; row++){
        for (int col = 0; col < N; col++){
            float sum = 0.0f;
            for (int i = 0; i < K; i++){
                sum += h_A[row * K + i] * h_B[i * N + col];
            }
            h_C_cpu[row * N + col] = sum;
        }
    }

    // 分配GPU内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // 将数据从CPU复制到GPU
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // GPU计算结果
    matMul(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 将结果从GPU复制到CPU
    cudaMemcpy(h_C_gpu, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Matrix A (%dx%d):\n", M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) printf("%5.1f", h_A[i * K + j]);
        printf("\n");
    }
    printf("\nMatrix B (%dx%d):\n", K, N);
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) printf("%5.1f", h_B[i * N + j]);
        printf("\n");
    }
    printf("\nGPU Result (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) printf("%5.1f", h_C_gpu[i * N + j]);
        printf("\n");
    }

    // 验证
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
    test_MatMul();
    return 0;
}



