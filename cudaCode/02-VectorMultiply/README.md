# 02-VectorMultiply

## 1.整体思路

给定MxK的矩阵A和KxN的矩阵B，乘积矩阵C为MxN，其中C[i][j] = Σ(A[i][k] * B[k][j])，k=0~K-1

## 2. 优化方向

1. 并行粒度设计: 每个线程负责计算C的一个元素(最直观的方式), 或多个元素(利用共享内存)
2. 内存访问模式: 矩阵A按行访问, 矩阵B按列访问, 以减少全局内存访问次数
3. 共享内存优化: 将A和B的子块加载到共享内存中, 减少全局内存访问次数(全局内存延迟高)
4. 线程块划分: 通常采用BLOCK_SIZE×BLOCK_SIZE的线程块（如 16×16 或 32×32），平衡并行度和资源占用


## Tiled Mulmul

使用`shared memory`优化矩阵乘法

**思路原理**

1. GPU上各种内存的访问速度为 `Global Memory << Shared Memory`

- Global Memory: 所有thread都可以访问, 对应到硬件上为DRAM, 并不在芯片上, 而是通过高带宽总线连接(DDR/GDDR/HBM)链接, 所以速度较慢
- Shared Memory: 每一个Block中的所有的thread都可共享。在芯片上, 对应到硬件上为SM内部的SRAM, 物理上是和L1 cache share的

2. Global Memory大而慢, Shared Memory小二快, 因此减少内存访问延迟的一个常见策略就是Tiling, 将数据分片加载到Shared Memory中, 减少全局内存访问次数

- 为了匹配thread, 每个sub matrix形如(BLOCK_SIZE x BLOCK_SIZE)
- 为了计算C中的sub matrix, 多次迭代load A/B中的sub matrix进入Shared Memory, 计算后将若干个结果进行叠加


