# 使用教程

## 导入必要的库

在开始之前，需要导入`functools.reduce`和`numpy`库。

```
...
```

## 创建实心球体掩膜

`create_solid_sphere`函数可以创建一个实心球体的掩膜，适用于三向等距的CT图像。

### 参数说明

- `radius`: 球体的半径，单位为体素。
- `center`: 球体中心的图像坐标，格式为`(x, y, z)`。
- `size`: CT图像的尺寸，格式为`(width, height, depth)`，与SimpleITK的`GetSize()`方法返回的结果相同。
- `fill_value`: 掩膜填充的数值，默认为`1`。

### 返回值

函数返回一个表示实心球体的三维NumPy数组，其坐标轴顺序与SimpleITK的`GetArrayFromImage()`方法相同。

### 示例代码

```python
# 定义球体参数
radius = 10
center = (15, 15, 15)
size = (30, 30, 30)

# 创建实心球体掩膜
sphere_mask = create_solid_sphere(radius, center, size)
```

## 合并掩膜

`combine_masks`函数用于合并多个形状相同的掩膜数组。

### 参数说明

- `masks`: 一个或多个NumPy数组，代表要合并的掩膜。

### 返回值

函数返回一个NumPy数组，表示合并后的掩膜。

### 示例代码

```python
# 创建两个掩膜
mask1 = create_solid_sphere(5, (10, 10, 10), size)
mask2 = create_solid_sphere(3, (20, 20, 20), size)

# 合并掩膜
combined_mask = combine_masks(mask1, mask2)
```

## 创建实心椭球体掩膜

`create_solid_ellipsoid`函数可以创建一个实心椭球体的掩膜。

### 参数说明

- `a`, `b`, `c`: 椭球在x, y, z轴方向的半轴长度，单位为体素。
- `center`: 椭球中心的图像坐标，格式为`(x, y, z)`。
- `size`: CT图像的尺寸，格式为`(width, height, depth)`，与SimpleITK的`GetSize()`方法返回的结果相同。
- `fill_value`: 掩膜填充的数值，默认为`1`。

### 返回值

函数返回一个表示实心椭球体的三维NumPy数组。

### 示例代码

```python
# 定义椭球参数
a, b, c = 10, 15, 20
center = (15, 15, 15)
size = (30, 30, 30)

# 创建实心椭球体掩膜
ellipsoid_mask = create_solid_ellipsoid(a, b, c, center, size)
```



## 对图像应用窗宽窗位调整

`apply_windowing`函数用于增强图像的对比度，通过调整窗宽窗位。

### 参数说明

- `image`: 输入的图像数组。
- `window_center`: 窗位的中心值。
- `window_width`: 窗宽的大小。
- `apply_limits`: 是否应用图像的最小和最大值限制。
- `preset`: 预设的窗宽窗位参数组合名称，默认为肺窗。

### 返回值

函数返回窗宽窗位调整后的图像数组。

### 示例代码

```python
# 输入图像数组
image = np.array([...], dtype=np.uint8)

# 应用窗宽窗位调整
windowed_image = apply_windowing(image, preset="lung")
```

## 评估结节是否位于肺部掩码内

`evaluate_nodule_inclusion`函数用于评估结节是否位于肺部掩码内，并可选地在CT图像上绘制结节。

### 参数说明

- `ct_slices`: CT图像的横断面数组。
- `lung_mask`: 与CT图像相对应的肺部掩码数组。
- `nodule_mask`: 结节掩码数组（可选）。
- `nodule_center_voxel`: 结节中心点的体素坐标（可选）。
- `nodule_radius_world`: 结节的世界坐标系半径（可选）。
- `ct_image`: CT图像，用于坐标转换（可选）。
- `ct_spacing`: 图像的体素间距（可选）。
- `display`: 是否在CT图像上显示结节。
- `inclusion_threshold`: 结节被肺部掩码包含的阈值。
- `save_images`: 是否保存绘制的结节图像。
- `save_dir`: 保存图像的目录路径。
- `ct_id`: CT的编号。

### 返回值

函数返回结节被肺部掩码包含的比例和布尔值，表示结节是否被肺部掩码包含。

### 示例代码

```python
# CT图像的横断面数组
ct_slices = np.array([...], dtype=np.uint8)

# 肺部掩码数组
lung_mask = np.array([...], dtype=np.uint8)

# 评估结节是否位于肺部掩码内
inclusion_ratio, is_included = evaluate_nodule_inclusion(
    ct_slices,
    lung_mask,
    display=True,
    save_images=True,
    save_dir="/path/to/save/images",
    ct_id="CT001"
)
```



## ct_seg_perf 函数

`ct_seg_perf` 函数用于评估CT图像分割的性能。

### 参数说明

- `pred_mask`: 预测的分割掩码，类型为 `np.ndarray`。
- `true_mask`: 真实的分割掩码，类型为 `np.ndarray`。
- `binarize`: 是否将掩码二值化，类型为 `bool`，默认为 `False`。
- `pred_mask_path`: 预测掩码的文件路径，类型为 `str`。
- `true_mask_path`: 真实掩码的文件路径，类型为 `str`。

### 返回值

函数返回一个字典，包含每个类别的Dice系数、IOU分数、准确率、精确率、召回率和F1分数。

### 示例代码

```python
# 使用示例
result = ct_seg_perf(
    pred_mask_path='path_to_pred_mask.npy',
    true_mask_path='path_to_true_mask.npy',
    binarize=True
)
```

## evaluate_lesion_inclusion 函数

`evaluate_lesion_inclusion` 函数用于评估结节是否位于指定器官掩码内，并可选地在CT图像上绘制结节。

### 参数说明

- `ct_source`: CT图像的来源，可以是文件路径、SimpleITK.Image对象或NumPy数组。
- `organ_source`: 器官掩码的来源，可以是文件路径、SimpleITK.Image对象或NumPy数组。
- `lesion_source`: 结节掩码的来源，可以是文件路径、SimpleITK.Image对象或NumPy数组。
- `binarize`: 是否将掩码二值化，类型为 `bool`，默认为 `True`。
- `display`: 是否在屏幕上显示结节绘制结果，类型为 `bool`，默认为 `False`。
- `inclusion_threshold`: 结节与器官掩码交集的阈值，用于判断结节是否被包含，类型为 `float`，默认为 `0.95`。
- `save_images`: 是否保存绘制的结节图像，类型为 `bool`，默认为 `False`。
- `save_dir`: 保存图像的目录路径，类型为 `str`，仅当 `save_images` 为 `True` 时需要。
- `ct_id`: CT的编号，用于命名保存的图像，类型为 `str`，仅当 `save_images` 为 `True` 时需要。

### 返回值

函数返回结节被器官掩码包含的比例（`inclusion_ratio`）和布尔值（`is_included`），表示结节是否被器官掩码包含。

### 示例代码

```python
# 使用示例
inclusion_ratio, is_included = evaluate_lesion_inclusion(
    ct_source='path_to_ct_image.nii',
    organ_source='path_to_organ_mask.nii',
    lesion_source='path_to_lesion_mask.nii',
    binarize=True,
    display=True,
    save_images=True,
    save_dir='/path/to/save/images',
    ct_id='CT001'
)
```
