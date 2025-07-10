# CMake学习

*Created by KennyS*

---


## 快速入门

1. 生成二进制文件 `add_executable`
2. 设置宏 `-DKey=Value`
3. 生成静态库与动态库 `add_library`
4. 引入第三方库 `find_package`
5. 其他命令


### 1. 生成二进制文件

- CMake固定文件名称 `CMakeLists.txt`
- 查看版本号 `cmake --version`
- 最简单的 `CMakeLists.txt`

    ```cmake
    cmake_minimum_required(VERSION 3.16)
    project(cmake_study)
    set(CMAKE_CXX_STANDARD 14)
    add_executable(main main.cpp)
    ```

### 2. 构建与编译

1. 常用构建系统

- `Makefile`
- `Ninja`：编译速度很快的工具

```cmake
cmake --help
cmake -S . -B build # -S 源文件  -B 二进制
cmake --build build

cmake -S . -B build -G Ninja
cd build && ninja
```

### 3. 添加子目录

- `add_subdirectory(hello)`

### 4. 设置源代码

- `AUX_SOURCE_DIRECTORY(src SRC_FILES)`

只能获取当前目录，无法获取子目录下的cpp文件


### 5. 指定编译模式

1. 常见的编译模式

- debug: 包含调试信息，不进行优化
- release：不包含调试信息，进行优化
- RelWithDebInfo:包含调试信息，进行优化
- MinSizeRel：尽可能减小生成文件的大小

```cmake
set(CMAKE_BUILD_TYPE Debug)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

- CMake默认编译的模式既不是`debug`也不是`release`


### 6. 设置参数默认值

- CMake变量对大小写不敏感，建议使用大写

```cmake
option(USE_PQ "Enable USE_PQ macro" OFF)
if (USE_PQ)
    add_compile_definitions(USE_PQ)
endif()
```



## C/C++项目快速运行

### find_package

1. `find_package(LZ4 REQUIRED)`

2. `find_package`从什么路径进行查找并导入

3. CMakeLists.txt中使用`find_package`导入依赖库，而本地安装在新的路径下，两者如何关联

- `find_package`默认查找路径: cmake安装路径下的module -> `/usr/share/cmake-3.16/Modules`

```cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}") // 当前cmake路径下的`cmake/`或者默认的`cmake/module/`
```


```cmake
// FindLz4.cmake
find_path(LZ4_INCLUDE_DIR lz4.h /opt/home/lz4/include ${CMAKE_SOURCE_DIR}/Module)
find_library(LZ4_LIBRARY NAMES lz4 PATHS /opt/home/lz4/lib ${CMAKE_SOURCE_DIR}/Module)

message(STATUS "enter cmake directory")
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    set(LZ4_FOUND TRUE)
endif (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
```