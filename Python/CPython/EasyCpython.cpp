/**
*****************************************************************************
*  Copyright (C), 2024, KennyS
*
*  @file    demo.cpp
*  @brief   CPython代码学习
            1. 调用python脚本的静态简单方式
            2. 动态加载python模块并执行函数
*
*  @author  KennyS
*  @date    2024-11-04
*  @version V1.1.0 20241104
*----------------------------------------------------------------------------
*  @note 历史版本  修改人员    修改日期      修改内容
*  @note v1.0      KennyS     2024-11-04    1.创建
*
*****************************************************************************
*/


#include <Python.h>
#include <iostream>

int main()
{   
    // Initital python
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        std::cout << "Python init fail!" << std::endl; 
        return 0;
    }

    // 把python代码作为字符串传给解释器执行
    // 实际场景为 C++向python传参，python返回结果
    // 导入模块
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");

    PyObject* pModule = PyImport_ImportModule("sayHello");
    if (!pModule)
    {
        PyErr_Print();
        return 1;
    }

    // 调用函数
    PyObject* pFunc = PyObject_GetAttrString(pModule, "say");
    if (!pFunc || !PyCallable_Check(pFunc))
    {
        PyErr_Print();
        return 0;
    }

    PyObject_CallObject(pFunc, NULL);

    // Free python
    Py_Finalize();

    return 0;
}