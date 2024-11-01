# Toys-Wedge

*Created by KennyS*

---

## 说明

尝试开始C++的一些工作


## Need

- `process_vtk.pyx`
- `setup.py`
- `main.cpp`
- `process_vtk.cpython-39-x86_64-linux-gnu.so`

## Build

```bash
python setup.py build_ext --inplace
export LD_LIBRARY_PATH=/data/renjuan/anaconda3/lib:$LD_LIBRARY_PATH
g++ -o main main.cpp -I/data/renjuan/anaconda3/envs/nnunet/include/python3.9 -lpython3.9 -L/data/renjuan/anaconda3/envs/nnunet/lib -L . -std=c++17 -ldl
./main
```

`lib include bin`



# 实际工控机没有装anaconda

## 需要

1. 虚拟环境envs：`wedge`

2. 需要复制 `wedge/bin include lib`

3. 环境变量

4. 设置`PYTHONPATH`: `CONDA_DIR/envs/torch/lib/pythonxx/site-packages`