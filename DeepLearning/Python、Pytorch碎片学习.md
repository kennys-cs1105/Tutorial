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
