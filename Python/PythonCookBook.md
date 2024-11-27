# PythonCookBook

*Created by KennyS*

---

## ch1 数据结构和算法

### 解压序列赋值给多个变量

**问题**：
现在有一个包含N个元素的元组或者是序列，怎样将里面的值解压后同时赋值给N个变量

**解决**：
1. 任何序列（或者可迭代对象）可以通过一个简单的赋值语句解压并赋值给多个变量。唯一的前提是变量的数量必须跟序列元素数量一致
    ```python
    data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
    name, shares, price, date = data
    ```

2. 如果变量个数和序列元素个数匹配不上
    ```python
    p = (4, 5)
    x, y, z = p
    
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    ValueError: need more than 2 values to unpack
    ```

**讨论**
1. 实际上这种解压赋值可以用在任何可迭代对象上，包括字符串，文件对象，迭代器，生成器
    ```python
    s = 'hello'
    a, b, c, d, e = s
    ```

2. 有时候只想解压一部分，可以使用任意的变量名去占位，但保证选用的变量名在其他地方没有被使用到
```python
data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
_, shares, price, _ = data
```

