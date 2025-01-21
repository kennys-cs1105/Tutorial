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

## CUDA计时

### 事件计时

1. 程序执行时间计时：是CUDA程序执行性能的重要表现
2. 使用CUDA事件（event）计时方式
3. CUDA事件计时可为主机代码、设备代码计时

### nvprof性能刨析

1. nvprof是一个可执行文件
2. 执行命令 `nvprof ./exe_name`

**Note**
1. `Warning: nvprof is not supported on devices with compute capability 8.0 and higher`：似乎是不支持高版本GPU


## 运行时GPU信息查询

### 运行时api查询GPU信息

1. 调用`cudaDeviceProp prop`

### 查询GPU计算核心数量

1. CUDA运行时API函数无法查询GPU核心数量
2. 根据GPU的计算能力进行查询


## 组织线程模型

### 数据存储方式

1. 数据在内存中是以线性、以行为主的方式存储
2. 16x8的二维数组中，在内存中一段连续的128个地址存储该数组

### 二维网格二维线程块

1. 二维网格和二维线程块对二维矩阵进行索引
2. 每个线程可负责一个矩阵元素的计算任务

![二维网格二维线程块](./asserts/二维网格线程块.PNG)

3. 线程与二维矩阵映射关系

$$
ix = threadIdx.x + blockIdx.x \times blockDim.x \\
iy = threadIdx.y + blockIdx.y \times blockDim.y
$$

4. 线程与二维矩阵映射关系

$$
idx = iy * nx + ix
$$


## GPU硬件资源

### 流多处理器-SM

1. GPU并行性靠流多处理器SM来完成（streaming multiprocessor）
2. 一个GPU由多个SM构成，Fermi架构SM关键资源如下
    - CUDA核心
    - 共享内存/L1缓存（shared memory/L1 cache）, 当前线程块中的线程都可以访问
    - 寄存器文件（RegisterFile）
    - 加载和存储单元（Load/Store Units）
    - 特殊函数单元（Special Function Unit）, 执行高效函数
    - Warps调度（Warps Scheduler）

3. GPU中每个SM都可以支持数百个线程并发执行
4. 以线程块block为单位，向SM分配线程块，多个线程块可被同时分配到一个可用的SM上
5. 当一个线程块被分配好SM后，就不可以再分配到其他SM上

![并发并行](./asserts/并发并行.PNG)

- 并行就是同时执行，两个任务互相不干扰
- 并发在一个核心中执行，任务在做高速切换

### 线程模型与物理结构

1. 线程模型可以定义成千上万个线程
2. 网格中所有线程块需要分配到SM上进行执行
3. 线程块内所有线程分配到同一个SM上执行，但是每个SM上可以被分配多个线程块
4. 线程块分配到SM中后，会以32个线程为一组进行分割，每个组成为一个wrap
5. 活跃的线程束数量会受到SM资源限制

### 线程束

1. CUDA采用单指令多线程SIMT架构管理执行线程，每32个为一组
2. 具体而言，一个线程块中，0-31个线程属于第0个线程束，32-63个线程属于第1个线程束，以此类推
3. 每个线程束中只能包含同一线程块中的线程
4. 每个线程束包含32个线程
5. 线程束是GPU硬件上真正做到了并行

## CUDA内存模型概述

### 内存结构层次特点

![内存结构层次](./asserts/内存结构层次.PNG)

1. 局部性原则
    - 时间局部性
    - 空间局部性

2. 底部存储器特点
    - 更低的每比特位平均成本
    - 更高的容量
    - 更高的延迟
    - 更低的处理器访问频率

3. CPU和GPU主存采用DRAM（动态随机存取存储器），低延迟的内存采用SRAM（静态随机存取存储器）

### CUDA内存模型

1. 模型
    - 寄存器register
    - 共享内存shared memory
    - 本地内存local memory
    - 常量内存constant memory
    - 纹理内存texture memory
    - 全局内存global memory

![CUDA内存模型](./asserts/CUDA内存模型.PNG)

2. CUDA内存和他们的主要特征
    - 物理位置
    - 访问权限
    - 可见范围
    - 生命周期

|内存类型|物理位置|访问权限|可见范围|生命周期|
|:---:|:---:|:---:|:---:|:---:|
|全局内存|芯片外|rw|所有线程和主机端|主机分配与释放|
|常量内存|芯片外|r|所有线程和主机端|主机分配与释放|
|纹理和表面内存|芯片外|r|所有线程和主机端|主机分配与释放|
|寄存器内存|芯片内|rw|单个线程|所在线程|
|局部内存|芯片外|rw|单个线程|所在线程|
|共享内存|芯片内|rw|单个线程块|所在线程块|


## 寄存器和本地内存

### 寄存器

1. 寄存器内存在片上（on-chip），具有GPU上最快的访问速度，但数量有限，属于GPU的稀缺资源
2. 寄存器仅在线程内可见，生命周期与所属线程一致
3. 核函数中定义的不加任何限定符的变量一般存放在寄存器中
4. 内建变量存放于寄存器中，例如`gridDim`, `blockDim`, `blockIdx`
5. 核函数中定义的不加任何限定符的数组有可能存在于寄存器中，但也有可能存在于本地内存中
6. 寄存器都是32位的，保存1个double类型的数据需要两个寄存器，寄存器保存在SM的寄存器文件
7. 计算能力5.9-9.0的GPU，每个SM中都是64K的寄存器数量，Fermi架构只有32K
8. 每个线程的最大寄存器数量是255个，Fermi架构是63个

### 本地内存

1. 寄存器放不下的内存会存放在本地内存
    - 索引值不能再编译时确定的数组存放于本地内存
    - 可能占用大量寄存器空间的较大本地结构体和数组
    - 任何不满足核函数寄存器限定条件的变量

2. 每个线程最多高达可使用512KB的本地内存
3. 本地内存从硬件角度看只是全局内存的一部分，延迟也很高，本地内存的过多使用会降低程序的性能
4. 对于计算能力2.0以上的设备，本地内存的数据存储在每个SM的一级缓存和设备的二级缓存中

### 寄存器溢出

1. 核函数所需的寄存器数量超出硬件设备支持，数据则会保存到本地内存（local memory）中
    - 一个SM运行并行多个线程块、线程束，总的需求寄存器容量大于64KB
    - 单个线程运行所需寄存器数量255个
2. 寄存器溢出会降低程序运行性能
    - 本地内存只是全局内存的一部分，延迟较高
    - 寄存器溢出的部分也可进入GPU的缓存中


## 共享内存

### 共享内存作用

1. 共享内存在片上（on-chip），与本地内存和全局内存相比具有更高的带宽和更低的延迟
2. 共享内存中的数据在线程块内所有线程可见，可用线程间通信，共享内存的生命周期也与所属线程块一致
3. 使用`__shared__`修饰的变量存放于共享内存中，共享内存可定义动态与静态两种
4. 每个SM的共享内存数量是一定的，也就是说如果在单个线程块中分配过度的共享内存，将会限制活跃线程束的数量
5. 访问共享内存必须加入同步机制：线程块内同步`void __syncthreads();`

### 共享内存作用

1. 不同计算能力的架构，每个SM中拥有的共享内存大小是不同的
2. 每个线程块使用的最大数量不同架构是不同的，计算能力8.9是100K

### 共享内存作用

1. 经常访问的数据由全局内存（global memory）搬移到共享内存（shared memory），提高访问效率
2. 改变全局内存访问内存的内存方式，提高数据访问的带宽

### 静态共享内存

1. 共享内存变量修饰符`__shared__`
2. 静态共享内存声明`__shared__ float tile[size, size];`
3. 静态共享内存作用域：
    - 核函数中声明，静态共享内存作用域局限在这个核函数中
    - 文件核函数外声明，静态共享内存作用域对所有核函数有效
4. 静态共享内存在编译时就要确定内存大小

### 共享内存和一级缓存划分

1. 在L1缓存和共享内存使用相同硬件资源的设备上，可通过`cudaFuncSetCacheConfig`运行时API指定设置首选缓存配置
2. func必须是声明为`__global__`的函数
3. 在L1缓存和共享内存大小固定的设备上，此设置不起任何作用


## 全局内存

### 全局内存

1. 全局内存在片外，容量最大，延迟最大，使用最多
2. 全局内存中的数据所有线程可见，Host端可见，且具有与程序相同的生命周期

### 全局内存初始化

1. 动态全局内存：主机代码使用CUDA运行时，API`cudaMalloc`动态声明内存空间，由`cudaFree`释放全局内存
2. 静态全局内存：使用`__device__`关键字静态声明全局内存

### 常量内存

1.  有常量缓存的全局内存，大小为64KB，访问速度较快
2. 常量内存中的数据对同一编译单元内所有线程可见
3. 使用`__constant__`修饰，不能定义在核函数中，并且静态定义
4. 可读，不可写
5. 给核函数传递数值参数时，这个变量存放于常量内存中
6. 必须在主机端使用`cudaMemcpyToSymbol`进行初始化


### GPU缓存

![GPU缓存](./asserts/GPU缓存.PNG)

1. 缓存种类
    - L1缓存
    - L2缓存
    - 只读常量缓存
    - 只读纹理缓存
2. 缓存作用
    - GPU缓存不可编程
    - 每个SM都有一个一级缓存，所有SM共享一个二级缓存
    - L1和L2用来存储本地内存和全局内存的数据，包括寄存器溢出的部分
    - GPU上只有内存加载可以被缓存，内存存储操作不能被缓存
    - 每个SM有一个只读常量缓存和只读纹理缓存，用于在设备内存中提高各自内存空间内的读取性能
3. L1缓存查询与设置
    - GPU全局内存是否支持L1缓存`cudaDeviceProp::globalL1CacheSupported`
    - 默认情况下数据不会缓存在L1/纹理缓存中，但可以通过编译指令启用
        - `-Xptxas -dlcm=ca`：除了带有禁用缓存修饰符的内联汇编修饰的数据外，所有读取都将被缓存
        - `-Xptxas -fscm=ca`：所有数据读取都被缓存


### 计算资源分配

1. 线程执行资源分配
    - 线程束本地执行上下文主要资源组成
        - 程序计数器
        - 寄存器
        - 共享内存
    - SM处理的每个线程束计算所需的计算资源属于片上资源(on-chip)，因此从一个执行上下文切换到另一个执行上下文是没有时间损耗的
    - 对于一个给定的内核，同时存在于同一个SM中的线程块和线程束数量取决于在SM中可用的内核所需寄存器和共享内存数量
2. 寄存器对线程数目的影响
    - 每个线程消耗的寄存器越多，则可以放在一个SM中的线程束就越少
    - 如果减少内核消耗寄存器的数量，SM便可以同时处理更多的线程束
3. 共享内存对线程块数量的影响
    - 一个线程块消耗的共享内存越多，则可以放在一个SM中的线程块就越少
    - 如果每个线程块使用的共享内存数量变少，SM便可以同时处理更多的线程块
4. SM占有率
    - 当计算资源分配给线程块时，线程块被称为活跃的块，线程块所包含的线程束被称为活跃的线程束，分为以下三种
        - 选定的线程束
        - 阻塞的线程束
        - 符合条件的线程束
    - 占有率是每个SM中活跃的线程束占最大线程束的比值
    - 网格和线程块大小的准则
        - 保持每个线程块中线程数量是线程束大小的倍数
        - 线程块不要设计的太小
        - 根据内核资源调整线程块的大小
        - 线程块的数量要远远大于SM的数量，保证设备有足够的并行

### 延迟隐藏

1. 概念
    - 指令延迟：指令发出和完成之间的时钟周期
    - 每个时钟周期中所有线程束调度器都有一个符合条件的线程束，可以达到计算资源的完全利用
    - GPU的指令延迟被其他线程束的计算隐藏称为延迟隐藏
    - 指令分为两种基本类型给
        - 算数指令
        - 内存指令

2. 算术指令隐藏
    - 从开始运算到得到计算结果的时钟周期，通常为4个时钟周期
    - 线程束数量 = 延迟 × 吞吐量
    - 吞吐量是SM中每个时钟周期的操作数量确定的
        - 16bit所需线程束数量 = 512 / 16 = 32

3. 内存指令隐藏
    - 从命令出发到数据到达目的地的时钟周期，通常为400-800个时钟周期