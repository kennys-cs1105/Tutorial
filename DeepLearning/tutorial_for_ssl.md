# Semi-Supervised Learning

*Created by KennyS*

---

##  参考

[FLARE22竞赛官网](https://flare22.grand-challenge.org/awards/)
[马骏Github](https://github.com/JunMa11/FLARE)

---


## 第一名方案

[仓库](https://github.com/Ziyan-Huang/FLARE22?tab=readme-ov-file)
[方案](https://zhuanlan.zhihu.com/p/657611778)

### 方案

1. 标注数据训练模型A
    - 50 Labeled Data
    -  training big nnUNet-A

2. 模型A给无标签数据打上伪标签(predict)
    - using big nnUNet, generate Pseudo Labels for 2000 Unlabeled Data

3. 伪标签 + 标注数据对模型A进一步优化得到最优模型B
    - using 50 + 2000, training nnUNet-B

4. *过滤低质量伪标签*

5. 标注数据 + 模型B生成的伪标签对小模型进行训练.
    - using 50 + 1924(selected pseudo labeled data)
    - training nnUNet-C

### 总结

1. 和现在的处理方式很接近
2. 有限标注训练小模型，推理大量无标签数据得到伪标签
3. 真实标注结合伪标签再训练
4. 我们的方案中没有伪标签评判这个环节
5. 训练开销比较大，先不处理

---


## 交大方案

[论文](https://link.springer.com/chapter/10.1007/978-3-031-23911-3_1)
[仓库](https://github.com/Shanghai-Aitrox-Technology/EfficientSegLearning)

### 思路

![交大方案](./asserts/交大方案.PNG#pic_center)

1. 数据预处理, 少量的标注文件 + 大量无标注文件
    - crop
    - seg_lung

2. 全监督分割: 配置full.yaml, 得到full.model
    - coarse_seg
    - fine_seg

3. 基于full.model推理
    - 基于训练得到的full.model推理未标注文件, 得到伪标签pseudo_label

4. 伪标签可靠性筛查

5. 基于标注文件 + 可靠伪标签文件配置semi.yaml, 训练semi.model
    - coarse_seg
    - fine_seg
