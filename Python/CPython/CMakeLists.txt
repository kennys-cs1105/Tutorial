# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 项目名称和 C++ 标准
project(CPython LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

#set(ENV{LD_LIBRARY_PATH} "/home/kennys/miniconda3/envs/torch/lib:$ENV{LD_LIBRARY_PATH}")
set(ENV{PYTHONPATH} "/home/kennys/miniconda3/envs/torch/lib/python3.10/site-packages:${CMAKE_SOURCE_DIR}")

# 指定可执行文件的名称和源文件
add_executable(main BuildValue.cpp)

# 指定 Python 头文件的路径
target_include_directories(main PRIVATE /home/kennys/miniconda3/envs/torch/include/python3.10)

# 指定链接库的路径
target_link_directories(main PRIVATE /home/kennys/miniconda3/envs/torch/lib)

# 链接需要的库
target_link_libraries(main PRIVATE python3.10 dl)
