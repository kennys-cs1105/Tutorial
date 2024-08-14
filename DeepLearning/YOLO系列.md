# YOLO系列

*Created by KennyS*

---

- [B站YOLO](https://space.bilibili.com/286900343)

- [YOLO series](https://github.com/z1069614715/objectdetection_script/tree/master)


## 魔改YOLO

### 添加注意力机制

1. 需要

- YOLOv8
- MHSA注意力

2. 配置yaml

- 在`models/nn/`路径内添加`MHSA.py`, 其中类名也是`MHSA`
- 在`task.py`文件中寻找`parse_model`函数, 在模块传递中进行添加

  ```python
  # 添加注意力
  elif m in {MHSA}:
      args = [ch[f], *args]
  ```
- 新建yolov8n_att.yaml配置表

  ```yaml
  # 记录原来层数的编号
  # 添加层之后编号变化
  # 添加层名字和类名相关

  # Parameters
  nc: 80 # number of classes
    # [depth, width, max_channels]
    n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs

  # YOLOv8.0n backbone
  backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]] # 9

    - [-1, 1, MHSA, [14, 14, 4]] # 10  -> channel和上一层有关, 添加其他参数即可, default:14 14 4

  # YOLOv8.0n head
  head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11
    - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # 12
    - [-1, 3, C2f, [512]] # 12                     # 13

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
    - [[-1, 4], 1, Concat, [1]] # cat backbone P3  # 15
    - [-1, 3, C2f, [256]] # 15 (P3/8-small)        # 16

    - [-1, 1, Conv, [256, 3, 2]]                   # 17
    - [[-1, 13], 1, Concat, [1]] # cat head P4     # 18
    - [-1, 3, C2f, [512]] # 18 (P4/16-medium)      # 19

    - [-1, 1, Conv, [512, 3, 2]]                   # 20
    - [[-1, 10], 1, Concat, [1]] # cat head P5      # 21
    - [-1, 3, C2f, [1024]] # 21 (P5/32-large)      # 22

    - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
  ```
  



#### Note

1. 一些默认路径有误, 查询 `~/.config/Ultralytics`