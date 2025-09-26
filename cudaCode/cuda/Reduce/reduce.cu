#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

#include "buffer.h"

#define BLOCK_SIZE 256
#define ITERATIONS 100
#define ERROR_TOLERANCE 1e-5f


// Kernel funtion
__global__ void reduce_kernel_baseline(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();
    
    // 1. 块内归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid * (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2. 块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}


__global__ void reduce_kernel_warp_divergent(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内归约
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // 2. 块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}


__global__ void reduce_kernel_bank_conflict(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 2. 块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}

__global__ void warp_reduce(volatile float *sdata, int tid)
{
    if (BLOCK_SIZE >= 64)
        sdata[tid] += sdata[tid + 32];
    if (BLOCK_SIZE >= 32)
        sdata[tid] += sdata[tid + 16];
    if (BLOCK_SIZE >= 16)
        sdata[tid] += sdata[tid + 8];
    if (BLOCK_SIZE >= 8)
        sdata[tid] += sdata[tid + 4];
    if (BLOCK_SIZE >= 4)
        sdata[tid] += sdata[tid + 2];
    if (BLOCK_SIZE >= 2)
        sdata[tid] += sdata[tid + 1];
}


__global__ void reduce_kernel_warp(const float *data, const size_t n, float *result)
{
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = global_id >= n ? 0 : data[global_id];
    __syncthreads();

    // 1. 块内归约 s>32时 多warp活跃 需要同步
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 1.1 块内归约 s<32 1个warp活跃 无需同步
    if (tid < 32)
    {
        warp_reduce(sdata, tid);
    }

    // 2. 块间归约
    if (tid == 0)
    {
        atomicAdd(&result[0], sdata[0]);
    }
}



