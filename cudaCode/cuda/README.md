# CudaCode

*Created By Kennys*

---

CUDA算子优化项目。实现优化常见的GPU计算算子。

## 算子介绍

1. Reduce: 实现高效的归约操作，包括求和、最大值等
2. GEMM: 通用矩阵乘法优化实现
3. Transpose: 矩阵转置算子, 重点优化内存访问模式
4. Softmax: 实现高效的softmax算子, 优化内存访问和并行计算

## 构建

```
cmake -B build
cmake --build build
```