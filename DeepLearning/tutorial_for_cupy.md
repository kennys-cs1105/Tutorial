# CUPY

*Created by KennyS*

---

## Introduction

1. numpy
    - 扩展语言库，支持大量的维度数组与矩阵运算
    - numpy函数是经过优化的，比直接编写函数快；并且使用矩阵、向量操作更快

    但上述numpy加速只是在cpu上实现，加速有限

2. cupy
    - 新的加速工具
    - 借助CUDA在GPU上实现numpy的库
    - 基于numpy数组的实现，GPU具有更多的CUDA核心，可以促成更好的并行加速
    - cupy接口是numpy的镜像，大部分情况下可以直接替换numpy使用，使用兼容的cupy代码替换numpy代码

3. 加速效果
    - 加速高度依赖于自身正在处理的数组大小
    - 数据点达到1000w，速度猛然提升；超过1亿，速度提升明显；numpy在1000w一下，实际运行更快
    - 需要检查GPU内存是否足以处理数据大小

4. 安装

```
pip install cupy-cuda12x
pip install cupy-cuda11x
```

cupy使用几乎和numpy完全一致，只需要`import cupy`，将对应的`numpy.xx`替换为`cupy.xx`

```python
import cupy as cp
x = cp.arange(6).reshape(2, 3).astype('f')
print(x, x.sum(axis=1))

>>> x = cp.arange(6, dtype='f').reshape(2, 3)
>>> y = cp.arange(3, dtype='f')
>>> kernel = cp.ElementwiseKernel(
...     'float32 x, float32 y', 'float32 z',
...     '''if (x - 2 > y) {
...       z = x * y;
...     } else {
...       z = x + y;
...     }''',
...     'my_kernel')
>>> kernel(x, y)
array([[ 0.,  2.,  4.],
       [ 0.,  4.,  10.]], dtype=float32)
```

5. 数据类型转换

- cupy与numpy转换
```python
import cupy as cp
import numpy as np

#cupy->numpy
numpy_data = cp.asnumpy(cupy_data)

#numpy->cupy
cupy_data = cp.asarray(numpy_data)
```

- cupy与torch转换
```python
# 需要借助中间库 dlpack，三者关系是：cupy.array<–>Dlpack.Tensor<–>torch.Tensor
from cupy.core.dlpack import toDlpack
from cupy.core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch

#pytorch->cupy
cupy_data = fromDlpack(to_dlpack(tensor_data))

#cupy->pytorch
tensor_data = from_dlpack(toDlpack(cupy_data))
```

- numpy与torch转换
```python
import numpy as np
import torch

#pytorch->numpy
numpy_data = tensor_data.numpy()

#numpy->pytorch
tensor_data = torch.from_numpy(numpy_data)
```

6. 多进程调试

```python
worker_pool_global = multiprocessing.Pool(processes=48)
```

7. 一致性测试

- np与cp在最后结果上会有细微差异（像素级），这个差异无法对上，但问题不大

8. 多卡训练

- 每次cupy操作对应的数据都必须在同一张卡上
- 要在一开始给cupy数据分配device，并在定义该device的语句下进行cupy操作


## Using

1. cp与np相互转换

```python
data1 = np.ndarray
# np转换为cp
data1_cp = cp.asarray(data1)

data2 = cp.ndarray
# cp转换为np
data2_np = cp.asnumpy(data2)
```

2. 使用时注意并发问题，基于CUDA运算一般要指定核心，并发可能一堆bug