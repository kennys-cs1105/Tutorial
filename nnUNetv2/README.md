# Tutorial for nnUNetv2

*Created by KennyS*

---

## OS

- Ubuntu22.04
- win10处理较麻烦


## ENV

1. python (最好3.9以上, 3.7/3.8可能出bug)
2. torch
3. nnunetv2: ```pip install -e .```  or  ```python3 setup.py install```
4. win10环境下:
```bash
# apex package
# https://github.com/NVIDIA/apex.git

pip install -v --no-cache-dir --global-option="–cpp_ext" --global-option="–cuda_ext" ./
```


## USING


### 数据路径

```nnUNet_raw```, ```nnUNet_preprocessed```, ```nnUNet_results```

1. nnUNet路径下创建/DATASET/nnUNet_raw

- 该文件夹为每个数据集创建子文件夹DatasetXXX_YYY（XXX是3位标识符，YYY是数据集名称
- 格式
├── dataset.json
├── imagesTr
│ ├── ...
├── imagesTs
│ ├── ...
└── labelsTr
├── ...
nnUNet_raw/Dataset002_NAME2
├── dataset.json
├── imagesTr
│ ├── ...
├── imagesTs
│ ├── ...
└── labelsTr
├── ...

2. nnUNet路径下创建/DATASET/nnUNet_preprocessed

- 保存预处理数据的文件夹, 训练过程中使用该文件夹下文件

3. nnUNet路径下创建/DATASET/nnUNet_results

- 保存模型权重, 如下载预训练模型, 则保存在该文件夹

4. 指定环境目录

- vim一个脚本, 避免在系统变量里定义

```bash
vim setPath.sh
#!/bin/bash
export nnUNet_raw="/media/nunet2_folder/nnUNet_raw"
export nnUNet_preprocessed="/media/nnunet2_folder/nnUNet_preprocessed"
export nnUNet_results="/media/nnunet2_folder/nnUNet_trained_models"
```

- 每次在终端中运行时, 需要```source setPath.sh```

- 验证环境变量: ```echo $nnUNet_results```

