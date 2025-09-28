# My-Torch-Extension

*Created By KennyS*

---

这是一个极简的、可拓展的Pytorch Extension，用于给Pytorch自定义后端算子. 我们提供了一个简单的加法算子，展示了如何通过pybind11将C++代码暴露接口给Python调用，同时也展示了如何通过setuptools对本project进行封装. 基于本project您可以更快地对比自定义的CUDA kernel与Pytorch自身的kernel的性能及功能.


## Build

```
python3 setup.py develop
```