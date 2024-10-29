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


## 线程全局索引计算方式

### 线程全局索引

- 一共有九种索引方式
    - 一维网格-对应一二三维线程块
    - 二维网格-对应一二三维线程块
    - 三维网格-对应一二三维线程块

1. 一维网格 一维线程块
    - 定义grid和block：
        ```
        dim3 grid_size(4);
        dim3 block_size(8);
        ```
    
    - 调用核函数
        ```
        kernel_fun<<<grid_size, block_size>>>(...);
        ```
    
    - 计算方式
        ```
        int id = blockIdx.x * blockDim.x + threadIdx.x
        ```

**Note**
- 索引类似于 `id=17=4*4+1` 

2. 二维网格 二维线程块
    - 定义grid和block：
        ```
        dim3 grid_size(2，2);
        dim3 block_size(4，4);
        ```
    
    - 调用核函数
        ```
        kernel_fun<<<grid_size, block_size>>>(...);
        ```
    
    - 计算方式
        ```
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        int id = blockId * (blockDim.x * blockDim.y) + threadId;
        ```

## NVCC编译流程与GPU计算能力

[cuda官方文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

### NVCC编译流程

1. nvcc分离全部源代码
    - 主机代码Host：C/C++语法
    - 设备代码Device：C/C++扩展语言

2. nvcc先将设备代码编译为PTX（Parallel Thread Execution）伪汇编代码，再将PTX代码编译为二进制的cubin目标代码

3. 在将源代码编译为PTX代码时，需要用选项`-arch=compute_XY`指定一个虚拟架构的计算能力，用以确定代码中能够使用的CUDA功能

4. 在将PTX代码编译为cubin代码时，需要用选项`-code=sm_ZW`指定一个真实架构的计算能力，用以确定可执行文件能够使用的GPU

### PTX

1. PTX是CUDA平台为基于GPU的通用计算而定义的虚拟机和指令集

2. nvcc编译命令总是使用两个体系结构
    - 虚拟的中间体结构
    - 实际的GPU体系结构

3. 虚拟架构更像是对应所需的GPU功能的声明

4. 虚拟架构应该尽可能选择低，适配更多实际的GPU；真实架构应该尽可能选择高，充分发挥GPU性能

### GPU架构与计算能力

1. GPU用于标识计算能力的版本号
    - 形式X.Y
    - X主版本号，Y次版本号


## CUDA程序兼容性问题

### 指定虚拟架构计算能力

1. C/C++源代码编译为PTX时，可以指定虚拟架构的计算能力，用来确定代码中能够使用的CUDA功能；这一步骤与GPU硬件无关

2. 编译指令，例如`nvcc helloworld.cu -o helloworld -arch=compute_61`，可执行文件`helloworld`只能在计算能力>=6.1的GPU上执行

### 指定真实架构计算能力

1. PTX指令转化为二进制cubin代码与具体的GPU架构有关

2. 二进制cubin代码大版本之间不兼容
3. 指定真实架构计算能力的时候必须指定虚拟架构计算能力
4. 指定的真实架构能力必须大于或等于虚拟架构能力
~~nvcc helloworld.cu -o helloworld -arch=compute_61 -code=sm_60~~
5. 真实架构可以实现低小版本到高小版本的兼容

### 指定多个GPU版本编译

1. 使得编译出来的可执行文件可以在多GPU中执行

2. 同时指定多组计算能力，编译选项`-gencode arch=compute_XY -code=sm_XY`
    - `-gencode arch=compute_35 -code=sm_35` 开普勒架构
    - `-gencode arch=compute_50 -code=sm_50` 麦克斯韦架构
    - `-gencode arch=compute_60 -code=sm_60` 帕斯卡架构
    - `-gencode arch=compute_70 -code=sm_70` 伏特架构
3. 编译出的可执行文件包含4个二进制版本，生成的可执行文件称为胖二进制文件`fatbinary`
4. 上述指令必须CUDA支持7.0计算能力

### nvcc即时编译

1. 在运行可执行文件时，从保留的PTX代码临时编译出cubin文件
2. 两个虚拟架构计算能力必须一致，例如
    ```
    -gencode=arch_compute_35, code=sm_35
    -gencode=arch_compute_50, code=sm_50
    ```
3. 简化`-arch=sm_XY`，等价于`-gencode=arch=compute_61,code=sm_61`

### nvcc编译默认计算能力

- cuda6.0之前：1.0
- cuda6.5-8.0：2.0
- cuda9.0-10.2：3.0
- cuda11.6：5.2


## CUDA矩阵加法运算程序

### CUDA程序基本框架

```cpp
#include <头文件>

__global__ void 函数名(param)
{
    kernel_func()
}
int main(void)
{
    // 1. 设置GPU设备
    setGPU(); 

    // 2.  分配主机和设备内存
    int iElemCount = 512; // 设置元素数量
    size_t stBytesCount = iElemCount * sizeof(float) // 字节数

    // 分配主机和设备内存 初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
    {
        memset(fpHost_A, 0, stBytesCount); // 初始化0
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    }
    初始化主机中的数据
    数据从主机复制到设备
    调用核函数在设备中计算
    将计算得到的数据从设备传到主机
    释放主机与设备内存
}
```

### 设置GPU设备

1. 获取GPU设备数量
    ```cpp
    int iDeviceCount = 0;
    cudaGetDeviceCount(&iDeviceCount);
    ```

2. 设置GPU执行时使用的设别
    ```cpp
    int iDev = 0;
    cudaSetDevice(iDev);
    ```

### 内存管理

1. CUDA通过内存分配、数据传递、内存初始化、内存释放进行内存管理

|STD C FUNC|CUDA C FUNC|
|:---:|:---:|
|malloc|cudaMalloc|
|memcpy|cudaMemcpy|
|memset|cudaMemset|
|free|cudaFree|

### 内存分配

1. 主机分配内存 `extern void *malloc(unsigned int num_bytes)`
    ```cpp
    float *fpHost_A;
    fpHost_A = (float *)malloc(nBytes);
    ```
2. 设备分配内存
    ```cpp
    float *fpDevice_A;
    cudaMalloc((float **) & fpDevice_A, nBytes);
    ```

### 数据拷贝

1. 主机数据拷贝`void * memcpy(void *dest, const void *src, size_t n)`
    ```cpp
    memcpy((void *)d, (void*)s, nBytes);
    ```

2. 设备数据拷贝
    ```cpp
    cudaMemcpy(Device_A, Host_A, nBytes, cudaMemcpyHostToHost)
    ```

**Note**
    ```cpp
    cudaMemcpyHostToHost // 主机 -> 主机
    cudaMemcpyHostToDevice // 主机 -> 设备
    cudaMemcpyDeviceToHost // 设备 -> 主机
    cudaMemcpyDeviceToDevice // 设备 -> 设备

    cudaMemcpyDefault // 默认
    ```

### 内存初始化

1. 主机内存初始化`void *memset(void *str, int c, size_t n)`
    ```cpp
    memset(fpHost_A, 0, nBytes);
    ```

2. 设备内存初始化
    ```cpp
    cudaMemset(fpDevice_A, 0, nBytes);
    ```

### 内存释放

1. 释放主机内存
    ```cpp
    free(pHost_A);
    ```

2. 释放设备内存
    ```cpp
    cudaFree(pDevice_A);
    ```

### 自定义设备函数

1. 设备函数
    - 定义只能执行在GPU设备上的函数
    - 设备函数只能被核函数或其他设备函数调用
    - 设备函数用`__device__`修饰

2. 核函数kernel function
    - 用`__global__`修饰，一般由主机调用，在设备中执行
    - `__global__`修饰符既不能和`__host__`同时使用，也不能与`__device__`同时使用

3. 主机函数
    - 主机端的普通C++函数可用`__host__`修饰
    - 对于主机端的函数，`__host__`修饰符可省略
    - 可以用`__host__`和`__device__`同时修饰一个函数减少冗余代码，编译器会针对主机和设备分别编译该函数

    ```cpp
    #include <stdio.h>
    #include "./tools/common.cuh"

    __global__ void addFromGPU(float *A, float *B, float *C, const int N)
    {
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        const int id = tid + bid * blockDim.x;

        C[id] = A[id] + B[id];

    }

    void initialData(float *addr, int elemCount)
    {
        for (int i = 0; i < elemCount; i++)
        {
            addr[i] = (float)(rand() & 0xFF) / 10.f;
        }
        return;
    }

    int main(void)
    {
        // 1、设置GPU设备
        setGPU();

        // 2、分配主机内存和设备内存，并初始化
        int iElemCount = 512;                               // 设置元素数量
        size_t stBytesCount = iElemCount * sizeof(float);   // 字节数

        // （1）分配主机内存，并初始化
        float *fpHost_A, *fpHost_B, *fpHost_C;
        fpHost_A = (float *)malloc(stBytesCount);
        fpHost_B = (float *)malloc(stBytesCount);
        fpHost_C = (float *)malloc(stBytesCount);
        if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
        {
            memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
            memset(fpHost_B, 0, stBytesCount);
            memset(fpHost_C, 0, stBytesCount);
        }
        else
        {
            printf("Fail to allocate host memory!\n");
            exit(-1);
        }

        // （2）分配设备内存，并初始化
        float *fpDevice_A, *fpDevice_B, *fpDevice_C;
        cudaMalloc((float**)&fpDevice_A, stBytesCount);
        cudaMalloc((float**)&fpDevice_B, stBytesCount);
        cudaMalloc((float**)&fpDevice_C, stBytesCount);
        if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
        {
            cudaMemset(fpDevice_A, 0, stBytesCount);  // 设备内存初始化为0
            cudaMemset(fpDevice_B, 0, stBytesCount);
            cudaMemset(fpDevice_C, 0, stBytesCount);
        }
        else
        {
            printf("fail to allocate memory\n");
            free(fpHost_A);
            free(fpHost_B);
            free(fpHost_C);
            exit(-1);
        }

        // 3、初始化主机中数据
        srand(666); // 设置随机种子
        initialData(fpHost_A, iElemCount);
        initialData(fpHost_B, iElemCount);

        // 4、数据从主机复制到设备
        cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice);
        cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);


        // 5、调用核函数在设备中进行计算
        dim3 block(32); // block一般定义为32的倍数
        dim3 grid(iElemCount / 32);

        addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);    // 调用核函数
        // cudaDeviceSynchronize();

        // 6、将计算得到的数据从设备传给主机
        cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);


        for (int i = 0; i < 10; i++)    // 打印
        {
            printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
        }

        // 7、释放主机与设备内存
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        cudaFree(fpDevice_A);
        cudaFree(fpDevice_B);
        cudaFree(fpDevice_C);

        cudaDeviceReset();
        return 0;
    }
    ```

## CUDA错误检查

### 运行时API错误代码

1. CUDA运行时API大多支持返回错误代码，返回值类型：`cudaError_t`
2. 运行时API成功执行，返回值为`cudaSuccess`
3. 运行时API返回的执行状态值是枚举变量

### 错误检查函数

1. 获取错误代码对应名称`cudaGetErrorName`
2. 获取错误代码描述信息`cudaGetErrorString`

#### 检查函数

1. 在调用CUDA运行API时，调用`ErrorCheck`函数进行包装
2. 参数`filename`一般使用`__FILE__`，参数`lineNumber`一般使用`___LINE__`

3. 错误函数返回运行API调用错误代码
    ```cpp
    cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber)
    {
        if (error_code != cudaSuccess)
        {
            printf("CUDA error: \r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n",
            error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
            return error_code;
        }
        return error_code;
    }
    ```

#### 检查核函数

1. 错误检测函数问题：不能捕捉调用核函数的相关错误
2. 捕捉调用核函数可能发生错误的方法
    ```cpp
    ErrorCheck(cudaGetLaskError(), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    ```

3. 核函数定义`__global__ void kernel_function(argument arg)`

    ```cpp
    #include <stdio.h>
    #include "../tools/common.cuh"

    __device__ float add(const float x, const float y)
    {
        return x + y;
    }

    __global__ void addFromGPU(float *A, float *B, float *C, const int N)
    {
        const int bid = blockIdx.x;
        const int tid = threadIdx.x;
        const int id = tid + bid * blockDim.x; 

        if (id >= N) return;
        C[id] = add(A[id], B[id]);
        
    }


    void initialData(float *addr, int elemCount)
    {
        for (int i = 0; i < elemCount; i++)
        {
            addr[i] = (float)(rand() & 0xFF) / 10.f;
        }
        return;
    }


    int main(void)
    {
        // 1、设置GPU设备
        setGPU();

        // 2、分配主机内存和设备内存，并初始化
        int iElemCount = 4096;                     // 设置元素数量
        size_t stBytesCount = iElemCount * sizeof(float); // 字节数
        
        // （1）分配主机内存，并初始化
        float *fpHost_A, *fpHost_B, *fpHost_C;
        fpHost_A = (float *)malloc(stBytesCount);
        fpHost_B = (float *)malloc(stBytesCount);
        fpHost_C = (float *)malloc(stBytesCount);
        if (fpHost_A != NULL && fpHost_B != NULL && fpHost_C != NULL)
        {
            memset(fpHost_A, 0, stBytesCount);  // 主机内存初始化为0
            memset(fpHost_B, 0, stBytesCount);
            memset(fpHost_C, 0, stBytesCount);
        
        }
        else
        {
            printf("Fail to allocate host memory!\n");
            exit(-1);
        }


        // （2）分配设备内存，并初始化
        float *fpDevice_A, *fpDevice_B, *fpDevice_C;
        cudaMalloc((float**)&fpDevice_A, stBytesCount);
        cudaMalloc((float**)&fpDevice_B, stBytesCount);
        cudaMalloc((float**)&fpDevice_C, stBytesCount);
        if (fpDevice_A != NULL && fpDevice_B != NULL && fpDevice_C != NULL)
        {
            cudaMemset(fpDevice_A, 0, stBytesCount);  // 设备内存初始化为0
            cudaMemset(fpDevice_B, 0, stBytesCount);
            cudaMemset(fpDevice_C, 0, stBytesCount);
        }
        else
        {
            printf("fail to allocate memory\n");
            free(fpHost_A);
            free(fpHost_B);
            free(fpHost_C);
            exit(-1);
        }

        // 3、初始化主机中数据
        srand(666); // 设置随机种子
        initialData(fpHost_A, iElemCount);
        initialData(fpHost_B, iElemCount);
        
        // 4、数据从主机复制到设备
        cudaMemcpy(fpDevice_A, fpHost_A, stBytesCount, cudaMemcpyHostToDevice); 
        cudaMemcpy(fpDevice_B, fpHost_B, stBytesCount, cudaMemcpyHostToDevice); 
        cudaMemcpy(fpDevice_C, fpHost_C, stBytesCount, cudaMemcpyHostToDevice);


        // 5、调用核函数在设备中进行计算
        dim3 block(2048);
        dim3 grid((iElemCount + block.x - 1) / 2048); 

        addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);    // 调用核函数
        ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
        ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);


        // 6、将计算得到的数据从设备传给主机
        cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost);
    

        for (int i = 0; i < 10; i++)    // 打印
        {
            printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i+1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
        }

        // 7、释放主机与设备内存
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        cudaFree(fpDevice_A);
        cudaFree(fpDevice_B);
        cudaFree(fpDevice_C);

        cudaDeviceReset();
        return 0;
    }
    ```

