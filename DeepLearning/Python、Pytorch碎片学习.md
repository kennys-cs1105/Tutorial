# Python、Pytorch碎片学习

*Created by KennyS*

---

## 命名空间、作用域和本地函数

1. 函数可以访问函数内部创建的变量，也可以访问函数外部更高（甚至全局）作用域的变量。Python 中描述变量作用域的另一种更具描述性的名称是命名空间。默认情况下，函数内分配的任何变量都分配给本地命名空间。本地命名空间是在调用函数时创建的，并立即由函数的参数填充。函数完成后，本地命名空间将被销毁（有一些例外情况不在本文的讨论范围内）。例如
    ```python
    def func():
        a = []
        for i in range(5):
            a.append(i)
    ```

    - 当调用func（）时会创建空列表a，添加元素，在函数退出时销毁a，但是当我们在外部声明a时
    ```python
    a = []
    def func():
        a = []
        for i in range(5):
            a.append(i)
    ```
    - 每次调用func（）都会修改列表a

    - 可以在函数范围之外分配变量，但必须使用global或nonlocal关键字声明
    ```python
    a = None
    def bind_a_variable():
        global a
        a = []
    bind_a_variable()
    ```

    - 通常不鼓励使用global关键字，如果使用了很多全局变量，表明需要使用类编程


## 函数返回多个值

1. 用简单的语法返回多个值
    ```python
    def f():
        a = 5
        b = 6
        c = 7
        return a, b, c
    a, b, c = f()
    # return_value = f()
    ```


## 函数是对象

1. 对字符串进行清理

- 使这个字符串列表统一并准备好进行分析，需要做很多事情：删除空格、删除标点符号，并标准化正确的大写字母。一种方法是使用内置字符串方法以及 re 标准库模块来处理正则表达式：
    ```python
    import re
    def clean(strings):
        result = []
        for value in strings:
            value = value.strip()
            value = re.sub("!#?", "", value)
            value = value.title() # 所有单词以大写开始
            result.append(value)
        return result
    ```

2. 另一种方法是列出要对特定字符串集应用的操作：
    ```python
    def remove_punc(value):
        return re.sub("!#$?", "", value)
    clean_ops = [str.strip, remove_punc, str.title]
    def clean_strings(strings, ops):
        result = []
        for value in strings:
            for func in ops:
                value = func(value)
            result.append(value)
        return result
    
    clean_strings(states, clean_ops)   
    ```

3. 将函数用作其他函数的参数，例如内置的 map 函数，它将函数应用于某种序列：
    ```python
    for x in map(remove_punc, states)
        print(x)
    ```


## Python编程习惯

1. 使用f-string，增加可读性，避免在复杂情况下出错
    ```python
    name = "Alice"
    greeting = f"Hello, {name}"
    ```

2. 使用上下文管理器，在代码块执行完毕后自动关闭文件，即便发生异常也会被正确处理，更加安全可靠
    ```python
    with open("example.txt", "r") as file:
        content = file.read()
    ```

3. 避免使用裸except，这会捕捉所有的异常，导致在处理异常时掩盖真正的错误
    ```python
    try:
        result = 10 / 0
    except ZeroDivistionError:
        print("Error occurred")
    ```

4. 避免默认参数使用可变对象。默认参数在函数定义时被计算，而不是函数调用时。默认参数如果是可变对象，那么在所有函数调用时被共享，可能导致意外的结果
    ```python
    def add_item(item, items=None):
        if items is None:
            items = []
        items.append(item)
        return items
    ```

5. 使用推导式，减少代码，提高效率，简洁，可读性高
    ```python
    squares = [i ** 2 for i in range(5)]
    ```

6. 使用isinstance，而非使用type检查类型，type无法处理继承关系，不够灵活
    ```python
    value = 42
    if isinstance(value, int):
        print("it is an integer")
    ```

7. 使用is，而不是==进行判断。
    ```python
    if x is None:
        print("x is None")
    ```

8. 避免使用bool或len进行条件检查，使用可读性更高的表达式，不必显式的检查长度或真值
    ```python
    my_list = [1, 2, 3]
    if my_list:
        print("List is not empty")
    ```

9. 使用enumerate，而不是range，enumerate在迭代中提供索引和值，比手动追踪索引更好
    ```python
    my_list = ["apple", "banana", "orange"]
    for i, item in enmuerate(my_list):
        print(i, item) # index从0开始
    ```

10. 使用items进行查找键值，避免额外的字典查找
    ```python
    my_dict = {"a":1, "b":2, "c":3}
    for key, value in my_dict.items():
        print(key, value)
    ```

11. 在生产环境中使用logging，而不是print
    ```python
    import logging
    reseult = calculate_result()
    logging.info("Result: %s", result)
    ```

12. 避免使用导入符号，可能导致命名冲突，不利于代码维护

    `from module import specific_function`

13. 使用原始字符串，提高代码可读性，尤其是处理路径、正则表达式等需要反斜杠的情况。原始字符串告诉读者不要解释反斜杠为转义字符
    ```python
    raw_string = r"d:\test\file.txt"
    # 避免
    regular_string = "d:\\test\\file.txt"
    ```


## pandas相关

1. 基于条件筛选行
    ```python
    import pandas as pd

    data = {'name': ['Alice', 'Bod', 'Charlice', 'David', 'Eva'],
            'Age': [24, 27, 22, 32, 29],
            'Score': [85, 90, 78, 88, 92]}
    df = pd.DataFrame(data)

    filtered_df = df[df['Age'] > 25]
    ```

2. 多条件筛选
    `filtered_df = df[(df['Age'] > 25) & (df['Score'] > 85)]`

3. isin()筛选
    `filtered_df = df[df['name'].isin(['Alice', 'Bod'])]`

4. 使用str.contains()
    `filtered_df = df[df['name'].str.contains('a', case=False)]`

5. query筛选
    `filtered_df = df.query('Age > 25')`

6. 筛选空值
    ```python
    df.loc[5] = ['Frank', None, 80]
    filtered_df = df[df['Age'].isna()]
    ```


## Python包

### 定义

1. 用来存放相关联的模块，一个包含有__init__.py文件的文件夹就可以被视作为一个包
2. __init__.py文件可以是空文件，也可以包含一些初始化代码

### __init__.py文件作用

1. 标识包：该目录被视作为一个包
2. 初始化代码：__init__.py文件中编写初始化代码，例如一些特定函数、类，设置全局变量等，当包第一次被导入时，这些代码会被执行
3. 控制导入行为：利用__init__.py文件控制包内的哪些内容对外可见

### 导入模块的3种方式

**基本概念**
1. 模块：包含python定义和语句的文件
2. 包：一个有层次结构的目录，包含`__init__.py`文件，该文件可以为空，包内可以包含其他模块或子包

**示例模块**

```python
# mymodule.py
def say_hello(name):
    return f"你好，{name}！欢迎来到山海摸鱼人的世界。"

def get_current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

1. 临时添加模块完整路径

```python
import sys
sys.path.append('/home/yourname/Documents/PythonCode')  # 添加路径
import mymodule

print(mymodule.say_hello('小白'))  # 同上
print(mymodule.get_current_time())  # 同上
```

2. 将模块保存到指定位置

`envs/lib/pythonxx/site-packages`

3. 设置环境变量，通过设置`PYTHONPATH`，可以让python自动查找并加载不在默认路径下的模块

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/module"
```

### 第三方库下载和安装

1. 常用pip命令
    - `pip install name`
    - `pip uninstall name`
    - `pip list`
    - `pip show name`
    - `pip freeze`：列出当前环境下的所有第三方库，通常用于生成依赖文件

2. 环境
    - `pip freeze > requirements.txt`
    - `pip install -r requirements.txt`


### 编程模式

**if __name__ == '__main__'**

1. 这个条件语句用于判断当前模块是否作为主程序运行。如果一个.py文件被直接执行，例如通过命令行输入`python script.py`，那么该文件中的`__name__`变量将被设置`'__main__'`。相反，如果该文件是被另一个脚本导入的话，`__name__`将会是模块的名字。

```python
# 文件名：main_example.py
def greet(name):
    """ 打印欢迎信息 """
    print(f"你好, {name}!")

def main():
    """ 主函数入口点 """
    name = "山海摸鱼人"
    greet(name)

if __name__ == '__main__':
    # 当此文件被直接运行时，执行main函数
    main()
```
2. 在这个例子中，我们定义了一个 greet 函数用来打印一条问候消息，并且定义了一个`main`函数作为程序的入口点。只有当`main_example.py`被直接运行时，才会调用`main()`函数；而如果我们尝试从其他文件导入它，则不会自动执行`main()`中的代码。

**应用场景**

1. 提高效率

- 使用`if __name__ == '__main__'`可以帮助开发者更好地组织代码，使代码更加模块化。这样做的好处之一是可以很容易地对单个组件进行测试而不必担心整个应用程序的复杂性。比如，开发一个数据分析工具，其中包含多个功能如数据清洗、统计分析等。可以把这些功能封装成不同的函数或类，然后在一个统一的`main`函数里调用它们。这样做不仅让代码更清晰易读，也方便了单元测试和维护。

2. 构建可重用库

- 当你想要创建一个可以被其他项目使用的Python库时，确保你的库能够在被导入时不执行任何不必要的操作是非常重要的。通过将所有初始化逻辑放在`if __name__ == '__main__'`块内，你可以保证这些逻辑只会在库作为主程序运行时才被执行。例如，考虑这样一个场景：你编写了一个处理图像的小工具，包括读取图片、调整大小等功能。为了方便其他人使用，你决定将其打包成一个库。此时，你可能希望提供一些示例代码来展示如何使用这个库。利用`if __name__ == '__main__'`结构，你可以在同一个文件中同时实现核心功能以及演示用途，而不用担心破坏了库的正常使用。

```python
# 文件名：image_tool.py
import os
from PIL import Image

def resize_image(image_path, output_path, size=(800, 600)):
    """ 调整图片大小并保存到指定路径 """
    with Image.open(image_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)
        print("图片已成功调整大小！")

if __name__ == '__main__':
    # 示例：调整一张图片的大小
    image_dir = '原始图片'
    output_dir = '调整后图片'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            input_path = os.path.join(image_dir, filename)
            output_path = os.path.join(output_dir, f"resized_{filename}")
            resize_image(input_path, output_path)
```

3. 安全性

- 将敏感操作限制在`if __name__ == '__main__'`内部还可以作为一种安全措施，防止某些不应该在导入时执行的操作被执行。

4. 调试与测试

- 对于大型项目来说，在开发过程中频繁地运行整个应用程序可能会非常耗时。通过这种方式，开发者可以直接针对特定部分进行调试或测试，从而加快开发速度。

## python异常处理

1. 重新引发异常
    - 如果当前层无法完全处理异常，应该在完成可能的操作后，将异常转译为更具体的、与当前层次相关的异常类型，并将其传递给上层调用者。这样可以让上层根据其上下文做出更合适的响应。

2. 在合适的层级处理异常
    - 如果当前代码块不知道如何处理某个异常，不应该捕获它。让该异常向上抛出，直到被一个知道如何处理它的层级所捕获
    - 避免空的`except`块，因为这会导致异常被隐藏，使得调试变得困难。至少应该记录异常的信息，以便于追踪问题所在
    - 不要仅仅打印异常信息而不做任何处理；应采取具体措施解决异常或者给出用户反馈

3. 分割 try 块
    - 将大块的`try`语句分割成多个小块，每个块只包含可能引发特定异常的代码
    - 这样可以针对不同的异常提供更精确的处理逻辑，同时简化了对异常之间关系的分析

4. 避免过于庞大的 try 块
    - 尽量减少`try`块中的代码量，以降低异常发生的可能性，并且更容易定位和修复问题
    - 对于可预见的情况，不要依赖异常处理机制；而是应当通过条件判断等常规手段预防错误的发生。

5. 异常不应用于流程控制
    - 异常处理机制设计初衷是为了处理非预期的错误情况，而非作为程序正常逻辑的一部分。
    - 不要为了流程控制而故意制造异常，这样不仅降低了程序效率，还增加了代码的复杂度。

6. 合适地使用异常
    - 只有对于那些外部的、不可预知的运行时错误才应该使用异常
    - 对于已知的、普通的错误，应该编写专门的错误处理代码，而不是简单地抛出异常

7. 成功异常处理的目标
    - 最小化代码混乱
    - 捕获并保留有用的诊断信息
    - 确保通知到合适的人或系统组件
    - 采用恰当的方式结束异常活动
    
### assert调试程序

1. 基本语法
```python
assert condition, message
```
    - `condition`：需要检查的布尔表达式
    - `message`：断言失败时要显示的信息

2. 例子
```python
def check_positive(num):
    # 确保输入数字大于0
    assert num > 0, "数值必须大于0"
    print(f"{snum} 是正数..")

# test
check_positive(5)  # 正常输出
# check_positive(-3)  # 解除注释后运行，将引发AssertionError
```

3. 应用场景

    - 定义数据结构
    ```python
    books = {
        'shu_ji_1': {'ming_cheng': '山海摸鱼人', 'zuo_zhe': '张三', 'nian_fen': 2020},
        'shu_ji_2': {'ming_cheng': '星辰大海', 'zuo_zhe': '李四', 'nian_fen': 2022}
    }
    ```

    - 实现
    ```python
    def add_book(id, info):
        """
        添加一本书到系统中。
        
        参数:
            id (str): 书籍唯一标识符
            info (dict): 包含书籍详细信息的字典
        
        返回:
            None
        """
        # 检查ID是否已存在
        assert id not in books, f"ID {id} 已经存在于系统中"
        
        # 检查必填项是否存在
        required_keys = ['ming_cheng', 'zuo_zhe']
        for key in required_keys:
            assert key in info, f"缺少必要信息: {key}"
        
        # 更新全局变量books
        books[id] = info
        print(f"成功添加了新书: {info['ming_cheng']}")

    # 测试添加书籍
    add_book('shu_ji_3', {'ming_cheng': '未来世界', 'zuo_zhe': '王五'})
    # add_book('shu_ji_2', {'ming_cheng': '重复测试'})  # 尝试使用已存在的ID添加，解除注释以查看效果
    ```

### 自定义异常类

1. 异常处理基础

```python
# 尝试执行可能出错的代码
try:
    # 这里尝试除以0，会触发ZeroDivisionError
    result = 10 / 0
except ZeroDivisionError as e:
    # 捕获到ZeroDivisionError时执行
    print(f"发生错误: {e}")
else:
    # 如果没有异常，则执行else块
    print(result)
finally:
    # 不管是否发生异常，都会执行finally块
    print("操作完成")
```

2. 为什么需要自定义异常类
    - 使异常具体化，更精确的描述问题所在

3. 自定义异常类

```python
class ShanshuiMoyurenError(Exception):
    """自定义异常：山海摸鱼人错误"""
    def __init__(self, message="默认错误信息"):
        self.message = message
        super().__init__(self.message)

# 使用示例
try:
    raise ShanshuiMoyurenError("发生了山海摸鱼人的特有错误！")
except ShanshuiMoyurenError as e:
    print(e)
```

4. try-except结构
```python
try:
    a = int(input("请输入第一个整数: "))
    b = int(input("请输入第二个整数: "))
    result = a / b
    print(f"{a} 除以 {b} 等于 {result}")
except ValueError:
    print("输入的不是整数！")
except ZeroDivisionError:
    print("除数不能为零！")
except Exception as e:
    print(f"发生了未知错误：{str(e)}")
```

5. try-except-else结构
    - `else`中的代码只有在`try`块没有引发异常的情况下才会执行，可以用于在没有错误发生时执行额外的逻辑

```python
try:
    a = int(input("请输入一个整数: "))
    b = int(input("请输入另一个整数: "))
    result = a + b
except ValueError:
    print("输入的不是整数！")
else:
    print(f"两数之和为 {result}")
```

6. try-except-finally结构
    - `finally`块中的代码总会被执行，无论前面的`try`或`except`块是否引发了异常，通常用于释放外部资源，如关闭文件或网络连接

```python
try:
    file = open("摸鱼.txt", "r")
    data = file.read()
except FileNotFoundError:
    print("文件未找到！")
except Exception as e:
    print(f"发生了未知错误：{str(e)}")
finally:
    # 尝试关闭文件，即使在前面的代码中出现了异常
    try:
        file.close()
    except:
        pass  # 如果文件从未被打开，那么忽略这个错误
```


### traceback

1. 异常三元素
    - type：异常类型，引发异常的类的引用，例如`FileNotFoundError`,`TypeError`
    - value：异常实例，详细信息
    - traceback：追踪回溯对象，包含异常发生时的调用栈信息，可以知道异常在哪个位置发生，以及调用了什么函数


## Cython

1. C/C++与python交互

### 在C++中使用python

1. 如果代码简单，最简单的方式
```cpp
#include <Python.h>

int main() {
    Py_Initialize();

    // 调用Python库中的matplotlib，绘制一幅简单的图形
    PyRun_SimpleString("import matplotlib.pyplot as plt\n"
                       "import numpy as np\n"
                       "x = np.linspace(0, 10, 100)\n"
                       "y = np.sin(x)\n"
                       "plt.plot(x, y)\n"
                       "plt.show()");

    Py_Finalize();
    return 0;
}
```

- 编译运行
```bash
export LD_LIBRARY_PATH=/home/kennys/miniconda3/envs/torch/lib:$LD_LIBRARY_PATH
g++ main.cpp -o main -I/home/kennys/miniconda3/envs/torch/include/python3.10 -L/home/kennys/miniconda3/envs/torch/lib -lpython3.10
./main
```

2. 使用Cython来封装python
    - 创建pyx文件来编写Cython代码，封装要使用的python库
    ```python
    # example.pyx

    cdef extern from "math.h":
        double sqrt(double x)

    def c_sqrt(double x):
        return sqrt(x)
    ```

    - 创建setup.py用于构建cython代码成为一个可被C++调用的扩展模块
    ```python
    # setup.py

    from distutils.core import setup
    from Cython.Build import cythonize

    setup(
        ext_modules=cythonize("example.pyx")
    )
    ```

    - 使用命令来构建并使用cython生成的扩展模块
    `python setup.py build_ext --inplace`

3. C++中使用python库
    - cython封装代码
    ```python
    # example.pyx

    cdef extern from "math.h":
        double sqrt(double x)

    def c_sqrt(double x):
        return sqrt(x)
    ```

    - 将封装代码编译成动态库
    `cythonize -i -3 example.pyx`

    - 生成一个名为`example.cpython-36m-darwin.so`的动态库，创建cpp文件调用
    ```cpp
    // main.cpp

    #include <iostream>
    #include <Python.h>

    int main() {
        Py_Initialize(); // 初始化Python解释器
        PyRun_SimpleString("import sys\nsys.path.append(\"/pylib\")"); // 添加Python库所在的路径
        PyObject* pModule = PyImport_ImportModule("example"); // 导入封装的Python库
        if (pModule) {
            PyObject* pFunc = PyObject_GetAttrString(pModule, "c_sqrt"); // 获取Python函数对象
            if (pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = PyTuple_Pack(1, PyFloat_FromDouble(25.0)); // 准备函数参数
                PyObject* pValue = PyObject_CallObject(pFunc, pArgs); // 调用Python函数
                double result = PyFloat_AsDouble(pValue); // 获取函数返回值
                std::cout << "Square root of 25: " << result << std::endl;
                Py_DECREF(pArgs); // 释放参数对象
                Py_DECREF(pValue); // 释放返回值对象
            }
            Py_DECREF(pFunc); // 释放函数对象
        }
        Py_DECREF(pModule); // 释放模块对象
        Py_Finalize(); // 释放Python解释器
        return 0;
    }
    ```

    - 编译
    `g++ -o main main.cpp -I/pylib -L/pylib -lpython3.6m`

4. 调用python库中的函数
    - 定义简单函数
    ```python
    # example.py
    def add(a, b):
        return a + b
    ```

    - 创建cpp文件调用
    ```cpp
    // main.cpp

    #include <Python.h>

    int main() {
        Py_Initialize(); // 初始化Python解释器

        // 导入Python脚本
        PyObject* pModule = PyImport_ImportModule("example");
        if (pModule) {
            // 获取Python函数对象
            PyObject* pFunc = PyObject_GetAttrString(pModule, "add");
            if (pFunc && PyCallable_Check(pFunc)) {
                // 准备函数参数
                PyObject* pArgs = PyTuple_Pack(2, PyLong_FromLong(3), PyLong_FromLong(4));
                // 调用Python函数
                PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
                // 获取函数返回值
                long result = PyLong_AsLong(pValue);
                // 输出返回值
                printf("Result of add function: %ld\n", result);
                // 释放参数对象
                Py_DECREF(pArgs);
                // 释放返回值对象
                Py_DECREF(pValue);
            }
            // 释放函数对象
            Py_XDECREF(pFunc);
            // 释放模块对象
            Py_DECREF(pModule);
        }
        // 释放Python解释器
        Py_Finalize();

        return 0;
    }
    ```

    - 编译
    `g++ -o main main.cpp -I/pylib -L/pylib -lpython3.7`


## pybind11

### 简单介绍

1. 实现功能，导出模块
    - 实现简单加法函数，将`add`方法放入`example`模块
```cpp
// src/example.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;
int add(int i, int j) {    
    return i + j;
}

PYBIND11_MODULE(example, m) {    
    m.doc() = "pybind11 示例"; // 模块文档字符串    
    m.def("add", &add, "一个简单的加法函数");
}
```

2. 打包成python包
```python
import sys
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [    
    Pybind11Extension("example",        
    ["src/example.cpp"],        
    define_macros = [('VERSION_INFO', __version__)],        
    ),
]

setup(    
    name="example",    
    version=__version__,    
    author="neeky",    
    author_email="neeky@live.com",    
    description="A test project using pybind11",    
    long_description="",    
    ext_modules=ext_modules,    
    extras_require={"test": "pytest"},    
    cmdclass={"build_ext": build_ext},    
    zip_safe=False,    
    python_requires=">=3.6",    
    install_requires=['pybind11==2.10.0']
)
```

3. 打包安装
```bash
python3 setup.py sdist

pip3 install dist/example-0.0.1.tar.gz
```