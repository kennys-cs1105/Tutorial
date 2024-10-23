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
    - 如果线程池数指定为<<<2,4>>>，则会打印8条Hello信息
