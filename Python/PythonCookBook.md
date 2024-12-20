# PythonCookBook

*Created by KennyS*

---

## ch1 数据结构和算法

### 1.1 解压序列赋值给多个变量

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


### 1.4 查找最大最小的N个元素

**解决**
1. 使用`hearq`模块，关键函数`nlargest`, `nsmallest`
    ```python
    import heapq
    nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
    print(heapq.nlargest(3, nums)) # Prints [42, 37, 23]
    print(heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]
    ```

2. 用于复杂数据结构
    ```python
    portfolio = [
        {'name': 'IBM', 'shares': 100, 'price': 91.1},
        {'name': 'AAPL', 'shares': 50, 'price': 543.22},
        {'name': 'FB', 'shares': 200, 'price': 21.09},
        {'name': 'HPQ', 'shares': 35, 'price': 31.75},
        {'name': 'YHOO', 'shares': 45, 'price': 16.35},
        {'name': 'ACME', 'shares': 75, 'price': 115.65}
    ]
    cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
    expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])

    # return 
    [{'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75}]
    ```

### 1.5 实现一个优先级队列

### 1.6 字典中的键映射多个值

**解决**
1. 一个字典就是一个键对应一个单值的映射。如果你想要一个键映射多个值，那么你就需要将这多个值放到另外的容器中， 比如列表或者集合里面
2. 使用`collection.defaultdict`来构造字典
3. `defaultdict`会自动为将要访问的键创建映射实例，即使它不存在；如果不需要这样的特性，可以手动创建`d.setdefault('b', [])`


### 1.7 字典排序

**解决**
1. 为了能控制一个字典中元素的顺序，你可以使用`collections`模块中的`OrderedDict`类。 在迭代操作的时候它会保持元素被插入时的顺序
    ```python
    from collections import OrderedDict

    d = OrderedDict()
    d['foo'] = 1
    d['bar'] = 2
    d['spam'] = 3
    d['grok'] = 4
    # Outputs "foo 1", "bar 2", "spam 3", "grok 4"
    for key in d:
        print(key, d[key])
    ```

2. 便于构建需要序列化或编码成其他格式的映射，例如json
    ```python
    import json
    json.dumps(d)
    # '{"foo": 1, "bar": 2, "spam": 3, "grok": 4}'
    ```

**注意**
1. `OrderedDict`内部维护了一个根据键插入顺序的双向链表，每当一个新的元素插入进来，它会被放到链表的尾部；对于一个已经存在键的重复赋值，不会改变键的顺序
2. `OrderedDict`的大小是普通字典的两倍，因为它内部维护着另一个链表，需要考虑内存消耗


### 1.8 字典运算

**解决**
1. 为了对字典值执行计算操作，通常使用`zip()`将键值反转
    ```python
    min_price = min(zip(prices.values(), prices.keys()))
    # min_price is (10.75, 'FB')
    max_price = max(zip(prices.values(), prices.keys()))
    # max_price is (612.78, 'AAPL')
    ```

2. 直接对字典运行，运算对象是字典的键而不是值，因此需要使用`zip()`
3. 不使用`zip()`，也可以对字典的值进行运算`min(prices.values())`
4. 需要得到键值的信息
```python
min(prices, key=lambda k: prices[k])
min_value = prices[min(prices, key=lambda k: prices[k])]
```

**注意**
1. `zip()`创建的是一个只能访问一次的迭代器，也就是说对`zip()`后的对象进行一次计算操作后，无法进行第二次计算操作
2. 在`zip()`转换为值键对后，如果有若干组值相等，执行计算操作后会根据键的大小返回相应的键值


### 1.9 查找两字典的相同点

**解决**
1. 基于`keys()`, `items()`执行集合操作
    ```python
    # Find keys in common
    a.keys() & b.keys() # { 'x', 'y' }
    # Find keys in a that are not in b
    a.keys() - b.keys() # { 'z' }
    # Find (key,value) pairs in common
    a.items() & b.items() # { ('y', 2) }
    ```

**注意**
1. 字典就是键集合和值集合的映射关系
2. 键视图的关键特性：支持集合操作
3. 字典的`items()`返回一个键-值对的元素视图对象，同样支持集合操作




## ch2 字符串和文本