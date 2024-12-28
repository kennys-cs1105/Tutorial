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

2. 同样适用于list set
    ```python
    list1 = list(set(a) - set(b))
    ```

**注意**
1. 字典就是键集合和值集合的映射关系
2. 键视图的关键特性：支持集合操作
3. 字典的`items()`返回一个键-值对的元素视图对象，同样支持集合操作


### 1.10 删除序列相同元素并保持顺序

**思考**
1. 删除相同元素自然而然想到集合(元素唯一性)

**解决**
1. 如果序列的值都是`hashable`, 利用集合或者生成器解决
    ```python
    def dedupe(items):
        seen = set()
        for item in items:
            if item not in seen:
                yield item
            seen.add(item)
    ```

2. 如果消除元素不可哈希, 例如dict类型中的重复元素
    - key指定了一个函数, 将序列元素转换成`hashable`
    - 例如
    
    ```
    a = [ {'x':1, 'y':2}, {'x':1, 'y':3}, {'x':1, 'y':2}, {'x':2, 'y':4}]
    list(dedupe(a, key=lambda d: (d['x'],d['y'])))
    ```

    ```python
    def dedupe(items, key=None):
        seen = set()
        for item in items:
            val = item if key is None else key(item)
            if item not in seen:
                yield item
                seen.add(item)
    ```

**注意**
1. 如果单纯的只是为了消除重复元素, 使用集合set就可以了, 但是set无法保证元素的顺序


### 1.11 命名切片

**讨论**

1. 示例代码
    ```python
    >>> items = [0, 1, 2, 3, 4, 5, 6]
    >>> a = slice(2, 4)
    >>> items[2:4]
    [2, 3]
    >>> items[a]
    [2, 3]
    >>> items[a] = [10,11]
    >>> items
    [0, 1, 10, 11, 4, 5, 6]
    >>> del items[a]
    >>> items
    [0, 1, 4, 5, 6]
    ```

2. 使用`slice`获取更多的信息
    ```
    >>> a = slice(5, 50, 2)
    >>> a.start
    5
    >>> a.stop
    50
    >>> a.step
    2
    >>>
    ```

3. 类似于获取指定的指针范围


### 1.12 序列中出现次数最多的元素

**解决**
1. `collections.Counter`
    ```python
    words = [
        'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
        'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
        'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
        'my', 'eyes', "you're", 'under'
    ]
    from collections import Counter
    word_counts = Counter(words)
    # 出现频率最高的3个单词
    top_three = word_counts.most_common(3)
    print(top_three)
    # Outputs [('eyes', 8), ('the', 5), ('look', 4)]
    ```

**讨论**
1. `Counter`接受任意的`hashable`元素构成的序列对象, return对象是一个字典 -> `"item": count_num`
2. 增加计数
    ```python
    >>> morewords = ['why','are','you','not','looking','in','my','eyes']
    >>> for word in morewords:
    ... word_counts[word] += 1
    ...
    >>> word_counts['eyes']
    9
    >>>

    # word_counts.update(morewords)
    ```
3. 可以和数学运算操作相结合
    ```python
    >>> a = Counter(words)
    >>> b = Counter(morewords)
    >>> a
    Counter({'eyes': 8, 'the': 5, 'look': 4, 'into': 3, 'my': 3, 'around': 2,
    "you're": 1, "don't": 1, 'under': 1, 'not': 1})
    >>> b
    Counter({'eyes': 1, 'looking': 1, 'are': 1, 'in': 1, 'not': 1, 'you': 1,
    'my': 1, 'why': 1})
    >>> # Combine counts
    >>> c = a + b
    >>> c
    Counter({'eyes': 9, 'the': 5, 'look': 4, 'my': 4, 'into': 3, 'not': 2,
    'around': 2, "you're": 1, "don't": 1, 'in': 1, 'why': 1,
    'looking': 1, 'are': 1, 'under': 1, 'you': 1})
    >>> # Subtract counts
    >>> d = a - b
    >>> d
    Counter({'eyes': 7, 'the': 5, 'look': 4, 'into': 3, 'my': 2, 'around': 2,
    "you're": 1, "don't": 1, 'under': 1})
    >>>
    ```

### 1.13 通过某个关键字排序一个字典列表

**解决**
1. `operator.itemgetter`

2. 根据字典的key进行排序
    ```python
    from operator import itemgetter
    rows_by_fname = sorted(rows, key=itemgetter('fname'))
    rows_by_uid = sorted(rows, key=itemgetter('uid'))
    print(rows_by_fname)
    print(rows_by_uid)
    ```

3. `itemgetter`支持多个keys
    ```python
    rows_by_lfname = sorted(rows, key=itemgetter('lname','fname'))
    print(rows_by_lfname)
    ```

**讨论**
1. 这个函数功能和`lambda`差不多, 但据说`itemgetter`运行快一点
    ```python
    rows_by_fname = sorted(rows, key=lambda r: r['fname'])
    rows_by_lfname = sorted(rows, key=lambda r: (r['lname'],r['fname']))
    ```


### 1.15 通过某个字段将记录分组


## ch2 字符串和文本