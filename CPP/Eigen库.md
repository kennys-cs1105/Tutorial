# C++ Eigen库学习

*Created by KennyS*

---

## 安装

1. `sudo apt-get install libeigen3-dev`

目录下存在`/usr/include/eigen3`, 不需要`sudo cp -r /usr/local/include/eigen3 /usr/include 
`

2. 源码安装

[官网](http://eigen.tuxfamily.org/index.php?title=Main_Page)
[文档](http://eigen.tuxfamily.org/dox/)

寻找版本，解压缩
```
tar -zxvf eigen-3.4.0.tar.gz
cd eigen-3.4.0/
mkdir build && cd build
cmake ..
sudo make install
sudo cp -r /usr/local/include/eigen3 /usr/include 
```

3. pangolin安装失败

在CMakeList中添加
```
# 添加头文件
include_directories("/usr/include/eigen3")
```

## 使用

### CMakeList

```
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
```

### 一些方法

1. 初始化

```cpp
MatrixxXd m = MatrixXd::Random(3,3) // 3*3的矩阵, 元素值在[-1,1]
MatrixxXd m = MatrixXd::Constant(3,3,2.4) // 3*3的常量矩阵, 元素值为2.4
Matrixx2d m = Matrix2d::Zero() // 2维的零矩阵
Matrix3d m3 = Matrix3d::Ones(); // 3维的1矩阵
Matrix4d m4 = Matrix4d::Identity(); // 4维的单位矩阵

Matrix3d m5 // 逗号初始化
m5 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
cout << "m0 =" << endl << m0 << endl;
cout << "m1 =" << endl << m1 << endl;
cout << "m2 =" << endl << m2 << endl;
cout << "m3 =" << endl << m3 << endl;
cout << "m4 =" << endl << m4 << endl;
cout << "m5 =" << endl << m5 << endl;
```

2. 调整矩阵大小

```cpp
MatrixXd m(2,5);
m.resize(4,3);
```

3. 一些简化名称

```cpp
typedef Matrix<float, 3, 1> Vector3f;
typedef Matrix<int, 1, 2> RowVector2i;
```

4. Array类

- 给Matrix的每一个元素都进行操作
- 需要Matrix和Array之间转换

```cpp
ArrayXXf a(3, 3);
ArrayXXf b(3, 3);
```

```cpp
m1.array()
a1.matrix()
```

5. 矩阵转置、共轭、共轭转置

```cpp
MatrixXcf a = MatrixXcf::Random(2,2);
cout << "Here is the matrix a\n" << a << endl;
cout << "Here is the matrix a^T\n" << a.transpose() << endl;
cout << "Here is the conjugate of a\n" << a.conjugate() << endl;
cout << "Here is the matrix a^*\n" << a.adjoint() << endl;
```

## 总结

1. 使用情况大致和numpy差不多, numpy里用的函数功能在eigen里基本上也都有相应函数实现, 查阅文档
2. eigen在静态数组运算上有优势, 动态数组考虑numpy或者cupy, 超大数据cupy优势明显