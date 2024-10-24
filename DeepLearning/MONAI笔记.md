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

## 一些构建教程

### 导入依赖

```python
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

print_config()
```

1. 其中`print_config()`可以查看当前MONAI需要的和未安装的第三方库版本信息

### MedMNIST数据集下载

```python
root_dir = "../MONAI_DATA_DIRECTORY"
print(root_dir)

resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
```

1. 设置随机种子`set_determinism(seed=0)`

2. 读取图像信息
    ```python
    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size

    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")
    ```

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

```python
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(), # 调整或增加输入数据的通道维度, 保证CNHW
        ScaleIntensity(), # 输入图像强度缩放到给定范围, 默认0-1, 类似伽马变换
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)

val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])
```

### 定义Dataset 网络 优化器

1. 和torch保持一致

```python
class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()
```

### 训练
```python
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
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