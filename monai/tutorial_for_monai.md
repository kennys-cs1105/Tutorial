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


## 数据载入和预处理

1. MONAI只处理nifti数据，其余格式需要转换
2. transform里的重采样间隔默认为(1.5,1.5,2)，但需要注意和nifti数据的xyz是否对上，需要检查
3. ScaleIntensityRanged，限制CT的HU值，并归一化到0-1，例如因为脾脏的HU值在(50,70)之间，所以(-57,164)以外的HU值就不考虑了
4. CropForegroundd功能慎用，它直接把你CT的空白部分去掉了，造成CT的size变化。如果你想让分割的结果再重叠到原CT，这个功能就禁用了
5. 注意train_transforms 和val_transforms 的超参数要一致
6. 不要一次性载入太多数据train_loader到你的GPU，进程会被killed
7. CropForegroundd和RandCropByPosNegLabeld组合，CropForegroundd会把空白区域裁掉，例如原size为(128,128,128), 裁切后为(110,120,100), 然后RandCropByPosNegLabeld会在这个尺寸上去ROI尺寸例如(96,96,96)进行patch裁切，需要注意裁切后的尺寸和ROI尺寸不要发生冲突。发生冲突也可以通过两种方式补救：首先可以在transform的最后补充ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128))。或者在dataloader中提供安全的collate函数collate_fn=pad_list_data_collate


### MONAI transforms

1. 通用接口
    - Transform // 各种Transorm继承这个类
    - MapTransform
    - Randomizable
    - Compose // 组合一系列的变换

2. 普通变换
    - 读取方式IO // LoadImage
    - 裁剪Crop&Pad // SpatialCrop
    - 强度变换Intensity // ScaleIntensity
    - 后处理Post-processing // LabelTocontour
    - 空间变换Spatial // Resize
    - 功能函数Utility // Totensor
    
3. 字典变换，和普通变换相似，在普通变换后面加一个`d`


### 数据集加载器

1. 常规Dataset
```
data_dir = ""
tranforms = Compose([...])
dataset = Dataset(data_dir, transforms)
sample = dataset[index]
```

2. CacheDataset: 预加载所有原始数据。将non-random-transform应用到数据并提前缓存，提高加载速度。如果数据集数量不是特别大，能够全部缓存到内存中，这是性能最高的Dataset。
```
transforms = Compose([...])
dataset = CacheDataset(data=train_dict, trainsforms, cache_rate=1.0,num_workers=4, progress=True)
```
- cache_rate: 缓存的百分比
- cache_num: 要缓存的项目数量
- num_workers: 线程数量
- progress: 进度条

3. PersistentDataset: 用于处理大型数据集。将数据应用non-random-transform，缓存在硬盘中。
```
transforms = Compose([...])
dataset = PersistentDataset(data=train_dict, transforms, cache_dir="./data/cache")
dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=torch.cuda.is_available())
```
- cache_dir: 数据加载到dataloader后，路径内不会有数据。当调用dataloader时才会有数据。
```
data = iter(dataloader).next()
```

### 测试
```python
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
```

## MONAI Model Zoo

### Lung nodule ct detection

1. MONAI集成的肺结节检测模型，`Lung_nodule_detection.zip`
2. 基于LUNA16数据集以及RetinaNet模型
    ```
    .
    ├── LICENSE
    ├── configs
    ├── datasets
    ├── docs
    ├── models
    └── scripts
    ```
3. MONAI Bundle的命令行
    ```bash
    python -m monai.bundle run --config_file configs/train.json # training
    python -m monai.bundle run --config_file configs/train.json --dataset_dir <actual dataset path> # using custom dataset
    python -m monai.bundle run --config_file configs/inference.json # inference
    ```
4. 准备`custom_dataset`以及`dataset.json`
5. trt模型暂不考虑

6. 运行`inference.json`时，准备的数据应当具备`image_meta_dict`，也就是元数据；没有元数据也可以从代码中修改，自己制定`meta_dict`，有点麻烦


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
