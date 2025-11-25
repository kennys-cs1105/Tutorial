# 01-VectorAddition

## 1.整体思路

利用GPU的并行计算能力，将向量加法的计算分布到多个线程中并行执行，从而提高计算效率。主要分为以下步骤：

1. GPU分配内存
2. 将CPU上的输入数据拷贝到GPU
3. 启动GPU核函数进行并行计算
4. 计算结果从GPU拷贝回CPU
5. 释放GPU内存


## 2. 线程索引计算

`threadIdx.x`: 当前线程在块内的索引 (0 ~ blockDim.x-1)
`blockIdx.x`: 当前块在网格内的索引 (0 ~ gridDim.x-1)
`blockDim.x`: 每个块中的线程数

