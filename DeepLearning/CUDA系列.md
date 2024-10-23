# CUDA基础

*Created by KennyS*

---

## GPU

1. GPU：图形处理器，对应CPU
    - GPU：数据运算
    - CPU：逻辑运算

2. 性能指标
    - 核心数
    - GPU显存
    - GPU计算峰值
    - 显存带宽

## CUDA编程语言

1. 基于C、C++，也可以基于python
2. 两种API
    - 驱动（driver）
    - 运行（runtime）

3. 查询GPU详细信息：`nvidia-smi -q`

## C++,CUDA

1. C++
    - 需要：g++
    - 编译：g++ hello.cpp -o hello

2. CUDA
    - 安装编译器：nvcc
    - nvcc支持纯C++代码编译
    - 编译扩展名为.cu的CUDA文件`nvcc hello.cu -o hello`
    ```cpp
    #include <stdio.h>
    int main(void)
    {
        printf("Hello world\n");
        return 0;
    }
    ```

## CUDA核函数

1. 核函数（Kernel function）
    - 核函数在GPU上进行并执行
    - 限定词`__global__`修饰，返回值必须是`void`
    ```cuda
    __global__ void kernel_function(argument arg)
    {
        printf(Hello world\n);
    }
    ```

2. 注意
    - 核函数只能访问GPU内存
    - 不能使用变长参数，明确参数个数
    - 不能使用静态变量
    - 不能使用函数指针
    - 具有异步性

3. CUDA程序编写流程
    ```
    int main(void)
    {
        主机代码 // GPU配置，数据处理
        核函数调用 // 并行加速
        主机代码 // 回传给主机，释放
        return 0；
    }
    ```
    - 注意：核函数不支持C++的iostream
    ```cuda
    #include <stdio.h>

    __global__ void hello_from_gpu()
    {
        printf("Hello World from the the GPU\n");
    }

    int main(void)
    {
        hello_from_gpu<<<1, 1>>>(); // 1个线程block, 1个线程
        cudaDeviceSynchronize(); // 同步函数, cpu与gpu执行效率不同
        return 0;
    }
    ```
    - 如果线程池数指定为`<<<2,4>>>`，则会打印8条Hello信息

## 线程模型

### 线程模型结构

1. 概念
    - 网格：grid
    - 线程块：block

2. 线程分块是逻辑上的划分，物理上线程不分块

3. 配置线程：`<<<grid_size, block_size>>>`

4. 最大允许线程块大小：1024；最大允许网格大小：$2^{31} - 1$，针对一维网络

**Note**
1. Device中包含多个网格，一个网格中包含多个线程块，一个线程块中包含多个线程

![线程模型](./asserts/线程模型.PNG#pic_center)


### 一维线程模型

1. 每个线程在核函数中都有唯一的身份标识

2. 每个线程的唯一标识由`<<<grid_size, block_size>>>`确定，`grid_size`，`block_size`保存在内建变量`build-in variable`，目前仅考虑一维的情况
    - `gridDim_x`：该变量数值等于执行配置中变量`grid_size`的值
    - `blockDim_x`：该变量数值等于执行配置中变量`block_size`的值
    - 这两个变量为固定大小

3. 线程索引保存成内建变量`build-in variable`
    - `blockldx.x`：指定一个线程在网格中的线程块索引值，范围$0 \sim gridDim.x-1$
    - `threadldx.x`：指定一个线程在线程块中的线程索引值，范围$0 \sim blockDim.x-1$

**Note**
1. 例如`<<<grid_size, block_size>>>` = `<<<2, 4>>>`，`blockldx.x`范围为$0 \sim 1$， `threadldx.x`范围为$0 \sim 3$

2. 例如`kernel_fun<<<2, 4>>>()`， `gridDim_x=2`，`blockDim_x=4`，线程唯一标识为`ldx = threadldx.x + blockldx.x * blockDim.x`

    ```cpp
    #include <stdio.h>

    __global__ void hello_from_gpu()
    {
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;

        const int id = threadIdx.x + blockIdx.x * blockDim.x;
        printf("Hello world from block %d and thread %d, global id %d\n", bid, tid, id);
    }

    int main(void)
    {
        hello_from_gpu<<<2,4>>>();
        cudaDeviceSynchronize();

        return 0;
    }
    ```

### 多维线程

1. CUDA可以组织三维的网格和线程块
2. `blockldx.x`和`threadldx.x`是类型为uint3的变量，类型为结构体，具有x，y，z成员（无符号类型）

$$
\left\{\begin{matrix}
blockIdx.x \\
blockIdx.y \\
blockIdx.z
\end{matrix}\right.
$$

### 推广到多维线程

1. 定义多维网格和线程块，基于构造函数

    ```cpp
    dim3 grid_size(Gx, Gy, Gz);
    dim3 block_size(Bx, By, Bz);
    ```

2. 定义一个$2 \times 2 \times 1$的网格，$5 \times 3 \times 1$的线程块

    ```cpp
    dim3 grid_size(2,2); // equal dim3 grid_size(2,2,1)
    dim3 block_size(5,3); // equal dim3 block_size(5,3,1)
    ```

    - 和矩阵维度相反，第一个为列

### 网格和线程块的限制条件

1. 网格大小限制
    - `gridDim.x`：$x^{31} - 1$
    - `gridDim.y`：$x^{16} - 1$
    - `gridDim.z`：$x^{16} - 1$

2. 线程块大小限制
    - `blockDim.x`：最大值1024
    - `blockDim.y`：最大值1024
    - `blockDim.z`：最大值64

**Note**
1. 线程块总的大小最大为1024
