# Tutorial for U-Mamba

*Created by KennyS*

---


## OS

Only in linux

- Ubuntu22.04


## ENV

1. python>=3.10
2. torch>=2.0.1
3. cuda11.8, 其他cuda版本可能会有问题

```bash
1. conda env: create -n umamba python=3.10 -y && conda activate umamba
2. torch: pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
3. causal_conv1d:
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
4. mamba: 
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
5. git repo: git clone https://github.com/bowang-lab/U-Mamba
6. cd U-Mamba/umamba &&  pip install -e .
```


## USING


### 环境变量设置

与nnUNetv2类似, 设置环境变量

1. setting
    - base = '/home/user_name/Documents/U-Mamba/data'
    - nnUNet_raw = join(base, 'nnUNet_raw')
    - nnUNet_preprocessed = join(base, 'nnUNet_preprocessed') 
    - nnUNet_results = join(base, 'nnUNet_results')


```bash
export nnUNet_raw="/data/kangshuai/yolo_luna/experiment/U-Mamba/data/nnUNet_raw"
export nnUNet_preprocessed="/data/kangshuai/yolo_luna/experiment/U-Mamba/data/nnUNet_preprocessed"
export nnUNet_results="/data/kangshuai/yolo_luna/experiment/U-Mamba/data/nnUNet_trained_models"
export CUDA_VISIBLE_DEVICES=1
```


### 数据准备

与nnUNetv2类似, 将用于nnUNetv2训练的数据迁移至数据路径下即可

```nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity```


### 模型

**Train 2d Models**

1. Train 2d U-Mamba_Bot Model

```nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaBot```

2. Train 2d U-Mamba_Enc Model

```nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerUMambaEnc```

**Train 3d Models**

1. Train 3d U-Mamba_Bot Model

```nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaBot```

2. Train 3d U-Mamba_Enc Model

```nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerUMambaEnc```

**Inference**

1. Predict with U-Mamba_Bot Model

```nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaBot --disable_tta```

2. Predict with U-Mamba_Enc Model

```nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c CONFIGURATION -f all -tr nnUNetTrainerUMambaEnc --disable_tta```

*Configuration: {2d, 3d_fullres}*


## Reference

1. Paper
```
@article{U-Mamba,
    title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
    author={Ma, Jun and Li, Feifei and Wang, Bo},
    journal={arXiv preprint arXiv:2401.04722},
    year={2024}
}
```

2. 修改参数: 与nnUNetv2不同, 可直接在U-Mamba代码中修改模型参数

3. Code

    - [U-Mamba](https://github.com/bowang-lab/U-Mamba/tree/main)
    - [Vision-Mamba](https://github.com/hustvl/Vim/tree/main)
    - [Mamba](https://github.com/state-spaces/mamba)
    - [nnUNet](https://github.com/MIC-DKFZ/nnUNet)


## Note

1. pip安装causal_conv1d时遇到问题, 下载源码编译

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal_conv1d
git checkout v1.0.2  # this is the highest compatible version allowed by Mamba
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
```

2. 训练```Mamba_Enc```模型会有问题, 只训练```Mamba_Bot```模型

3. 训练```Mamba_Bot```模型时, 指标先高再低至0, 调整学习率, 调至原lr的1/10
