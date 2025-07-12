# YOLO系列

*Created by KennyS*

---

- [B站YOLO](https://space.bilibili.com/286900343)

- [YOLO series](https://github.com/z1069614715/objectdetection_script/tree/master)

---


## YOLO相关

### 仓库

- [YOLOv8仓库](https://github.com/ultralytics/ultralytics)
- [YOLOv5仓库](https://github.com/ultralytics/yolov5)

### Paper

---


## YOLOv8

### 改进

1. **Backbone**
    - SPPF，相比于空间金字塔池化（SPP），加速
    - CSP梯度分流的思想，使用C2f替换C3，轻量化

2. **PAN-FPN(Neck)**
    - 使用PAN思想
    - PAN-FPN上采样阶段中的卷积结构删除
    - 使用C2f替换C3

3. **Anchor-Free**
    - 舍弃Anchor-Base

4. **损失函数**
    - 分类损失：VFL Loss
    - Bbox损失：DFL Loss + CIOU Loss

5. **样本匹配**
    - 舍弃IOU匹配或者单边比例的分配方式
    - 使用Task-Aligned-Assigner匹配方式

### 环境

- wsl
- python>=3.10，torch>=2.0
- pip install ultralytics

###  数据准备，训练自己的数据集

- 训练2d图像，bmp jpg png等格式（YOLO应该有要求）

1. 构建mydata数据集，路径下包含以下文件夹。
    - Annotations：VOC格式的xml标注信息
    - images：存放图像
    - labels：存放YOLO格式的txt标注信息
    - dataSet：训练、验证文件（train.txt，val.txt，记录文件路径）

- 注意：<font color='red'>*images和labels*</font>文件夹名字不可变（测试过），YOLO的api里写的是这两个名字，除非改api

2. 划分train.txt，val.txt（dataSet）代码：运行该代码，会在dataSet路径下生成train.txt，val.txt文件

    ```python
    """
    split_train_val.py
    """
    trainval_percent = 1.0
    train_percent = 0.9
    # xml_dir = opt.xml_path
    # dataSet_dir = opt.txt_path
    xml_files = os.listdir(xml_dir)
    if not os.path.exists(dataSet_dir):
        os.makedirs(dataSet_dir)
    
    num = len(xml_files)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)
    
    file_trainval = open(dataSet_dir + '/trainval.txt', 'w')
    file_test = open(dataSet_dir + '/test.txt', 'w')
    file_train = open(dataSet_dir + '/train.txt', 'w')
    file_val = open(dataSet_dir + '/val.txt', 'w')
    
    for i in list_index:
        name = os.path.join(images_dir, xml_files[i][:-4] + ".bmp") + '\n'
        if i in trainval:
            file_trainval.write(name)
            if i in train:
                file_train.write(name)
            else:
                file_val.write(name)
        else:
            file_test.write(name)
    
    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()
    ```

3. 标注转换：运行该代码，将VOC格式的xml标注转化为YOLO格式的txt标注（labels路径下）。<font color='red'>注意：classes中的类别需要参考xml文件中的`<name>`</font>

    ```python
    """
    数据转换
    voc_label.py
    """
    sets = ['train', 'val', 'test']
    classes = ["nodule"]   # 改成自己的类别
    
    def convert(size, box):
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h
    

    def convert_annotation(xml_file, txt_file):
        in_file = open(xml_file, 'r')  # 打开 XML 文件，'r' 表示读取模式
        out_file = open(txt_file, 'w')  # 打开输出的文本文件，'w' 表示写入模式
    
        tree = ET.parse(in_file)  # 使用 ElementTree 解析 XML 文件
        root = tree.getroot()  # 获取 XML 树的根节点
    
        size = root.find('size')  # 在根节点中找到 'size' 元素
        w = int(size.find('width').text)  # 获取图像宽度
        h = int(size.find('height').text)  # 获取图像高度
    
        # 遍历 XML 文件中的每个 'object' 元素
        for obj in root.iter('object'):
            difficult = 0  # difficult默认为0
            cls = obj.find('name').text  # 获取 'name' 元素的文本内容，即物体类别
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)  # 获取类别在类别列表中的索引
            xmlbox = obj.find('bndbox')  # 获取 'bndbox' 元素
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )  # 获取边界框坐标信息
    
            # 调用 convert 函数，将坐标信息转换为YOLO格式
            bb = convert((w, h), b)
    
            # 将转换后的信息写入输出文本文件
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
    
        in_file.close()  # 关闭输入文件
        out_file.close()  # 关闭输出文件
        
        xml_files = glob(os.path.join(xml_dir, "*.xml")) # 8685
    print(len(xml_files))

    for file in xml_files:
        print(file)
        convert_annotation(
            xml_file=file,
            txt_file=os.path.join(labels_dir, os.path.basename(file)[:-3] + 'txt')
        )
    ```

4. 配置文件
    - 配置数据文件mydata.yaml
    
    ```yaml
    train: "/home/kennys/experiment/Vessels/processing/cstl/pic/mydata/dataSet/train.txt"
    val: "/home/kennys/experiment/Vessels/processing/cstl/pic/mydata/dataSet/val.txt"
    # Classes
    names:
      0: nodule
    nc: 1
    ```

    - 修改YOLO仓库代码中的模型文件yolo.yaml（v8），类别数为nc=1

### 训练

1. CLI训练
    - 预训练权重需要自己下载
    - 在model处配置YOLO规模，会自动配置yolo.yaml
    - 第一次运行时，会下载yolov8n.pt模型用于测试精度
  
  ```bash
  yolo task=detect mode=train model=yolov8l.pt data=mydata.yaml epochs=500 batch=16 device=0
  ```

### 预测

1. CLI预测
    - 根据已有图像、训练权重进行预测

    ```bash
    yolo detect predict model=./runs/detect/train6/weights/best.pt \
        source=path \ 
        save_txt=True \
        save_conf=True
    ```

### 改进

---


## YOLO11

### 运行

1. 训练与预测的CLI代码和yolov8一致
2. 将模型、配置yaml修改为yolo11即可

### 改进

1. 将C2f改进为C3k2
    - 相比于C3，自定义卷积大小，使用灵活

2. 增加注意力模块C2PSA
    - 卷积提取特征，分为a，b两部分
    - b通过PSA Block进行自注意力计算，然后和a进行concat
    - 再用一个卷积提取特征

3. 修改检测头，使用深度可分离卷积替代传统卷积（降低参数量）

---


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


---


## 标签分配策略

### 最简单的标签分配

1. pic的标签类别为[x1, x2, x3, x4, x5]，其中target_A的类别为x2，模型network的预测为x_pred。对于target_A来说，需要计算CE_loss(x2, x_pred)

2. 对于语义分割任务（像素级分类任务）而言，图像为640x640，经过模型network输出后，得到640x640x5(class)的矩阵

### 目标检测中的标签分配

#### 定义

1. 对于YOLOv8，640*640的图像可以得到8400预测候选框
    - YOLOV8有三个尺度的输出：8倍下采样，16，32
    - 8400 = 80*80 + 40*40 + 20*20

2. 正样本：

---


## Note

1. 一些默认路径有误, 查询 `~/.config/Ultralytics`

2. 训练时如果loss出现NaN，可能是开启半精度的问题，也可能是所加模块不适用的问题
    - self.amp = False
  

## YOLO部署

- 部署时，为了提高效率，通常不直接使用训练生成的pt模型，而是把pt模型转化为可以通过tensorrt加速推理的engin文件
- yolo模型转化为tensorrt模型
- yolo模型转化为onnx模型

### 环境

- python==3.10
- torch==2.1.1
- cuda==11.8
- cudnn==8.9.3
- TensorRT==8.6.0.12

1. cuda安装
```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run
```

2. cudnn
```
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/include # 修改路径
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64
sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h 
sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*

# 查看版本
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

3. tensorrt安装
```
# tensorrt匹配cuda cudnn版本
export PATH=/home/hk/Package/TensorRT-8.6.1.6/bin:$PATH
export LD_LIBRARY_PATH=/home/hk/Package/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/hk/Package/TensorRT-8.6.1.6/lib:$LIBRARY_PATH
```

4. python安装
```
cd python
pip install tensorrt-py38...
```

### 代码

1. 以yolov8为例
    - [yolov8仓库](https://github.com/ultralytics/ultralytics)
    - [tensorrt](https://github.com/wang-xinyu/tensorrtx)

2. 构建
    - 生成wts文件
        - 修改`yolov8/gen_wts.py`中的模型路径
        - 生成`python3 gen_wts.py -w weight/yolov8n.pt -o weight/yolov8n.wts -t detect`
    - Build
        - 如果自己训练yolo，则可能需要修改yololayer.h中的超参，例如CLASS_NUM等
        - `yolov8.cpp`中调整`batch_size`
        - 构建
        ```
        mkdir build && cd build
        cp {yolov8}/weight/yolov8n.wts ./
        cmake ..
        make
        ```
        - 生成engine文件
            - 生成：`./yolov8_det -s yolov8n.wts yolov8n.engine n`
            - 推理：`./yolov8_det -d yolov8n.engine images/data g`

### 记录

1. 测试trt python安装报错`libnvinfer.so.8: cannot open shared object file: No such file or directory`

- 找不到tensorRT lib路径，需要修改`/etc/ld.so.conf`
```
sudo gedit /etc/ld.so.conf
# 添加 TensorRT lib 路径
/PATH/TO/TensorRT-7.1.3.4/lib
# 更新配置
sudo ldconfig
```

- 软链接错误，需要创建软链接
```
sudo ldconfig
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_adv_train.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_ops_train.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8 不是符号连接
/sbin/ldconfig.real: /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8 不是符号连接
```

2. 报错`/usr/bin/ld: warning: libopenjp2.so.7, needed by /usr/local/lib/libopencv_imgcodecs.so.4.5.5, not found (try using -rpath or -rpath-link)`

- 安装依赖
```
sudo apt-get update
sudo apt-get install libopenjp2-7
```
3. np.bool报错
- 在numpy1.20版本之后，修改为np.bool_