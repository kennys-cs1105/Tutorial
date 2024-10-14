# Tutorial for MONAI

*Created by KennyS*

---

## 参考

[MONAI官方文档](https://docs.monai.io/en/stable/)

---


## 版本相关

1. `ImportError: cannot import name 'AddChanneld' from 'monai.transforms'`
    - 似乎与版本无关，1.2.0与1.3.0同样报错
    - EnsureChannelFirst替换AddChanneld

2. 使用`LoadImage`和`LoadImaged`，而不是`LoadNiftid`

3. `applying transform <monai.transforms.utility.array.EnsureChannelFirst object at 0x7f2912417940>`

    - 似乎与numpy版本有关
    - numpy现版本1.26，降级为1.21.4，但无效

---


## Auto3dSeg
1. [仓库](https://github.com/lassoan/SlicerMONAIAuto3DSeg)

2. 权重下载
    - 仓库的release中选择权重
    - win目录下：`Users/username/.MONAIAuto3DSeg`

3. 3D Slicer插件安装
    - 插件管理（`Extension Manager`）安装`MONAIAuto3dSeg`
    - View菜单选择`Module Finder`选择安装的MONAI插件
    - 该插件需要安装torch第三方库
        - 自己下载torch.whl (2.4.1 cu118)
        - 在3D Slicer安装路径下寻找bin，路径下使用 `PythonSlicer.exe -m pip install 'path to whl'`
        - 选择ct图像，设置model，进行apply处理

4. 代码推理

    - 仓库源码中的Scripts
    - IO方式似乎有一些问题，原来的IO读取DICOM和NIFTI都无法保存预测结果
    - 根据预测的seg.array，修改了保存方式

---


## 构建Cache数据集

### 构建时需要注意的一些地方，参考MONAI文档就明白了

1. 数据集通常为一个字典，与nnUNetv1的数据格式相似
    ```json
    [{"image": "path", "label": "path"},
    .....]
    ```

2. 测试后发现构建数据集时，transform的相关函数都选用带d的，应该是默认对dict进行处理
    - 使用LoadImaged而不是LoadImage
    - 使用EnsureChannelFirstd而不是EnsureChannelFirst
    - 使用ScaleIntensityd而不是ScaleIntensity
    - 使用EnsureTyped而不是EnsureType
    - 其他的一些诸如resize和rotate等变换应用于dict时会报错，还没仔细研究

3. 对dict进行处理时，上述的变换函数中需要指定keys=["image", "label"]

### 构建CacheDataset

1. 具有缓存机制的 Dataset，可以在训练过程中加载数据并缓存确定性转换的结果

2. 缓存有内存限制，小规模数据可以使用，超过内存限制则会崩

3. 构建参考

```python
transforms = Compose([LoadImaged(),
                      EnsureChannelFirstd(),
                      Spacingd(),
                      Orientationd(),
                      ScaleIntensityRanged(),
                      RandCropByPosNegLabeld(),
                      ToTensord()])
```

### 构建PersistentDataset

1. CacheDataset每次构建都需要重新缓存一次，如果后续的代码崩了，则需要每次都浪费时间

2. PersistentDataset构建一个cache_dir用于存储数据，以便后续重复使用