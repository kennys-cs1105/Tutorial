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

---


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


### 训练

1. 预处理数据制作

```bash
nnUNetv2_train DATAnnUNetv2_plan_and_preprocess -h 可查看使用帮助
nnUNetv2_plan_and_preprocess -d 131（你的数据ID） --verify_dataset_integrity
```

2. 训练

```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [其他选项，参见 -h]
nnUNetv2_train 131 3d_fullres 1 (表示使用131这个数据集，模型用3d_fullres，训练第一折)
```

model:
 - 2d
 - 3d_fullres
 - 3d_lowres

中断继续训练:```nnUNetv2_train 131 3d_fullres 1 --c```

MSD数据集从nnUNetv1转换到v2: ```nnUNetv2_convert_MSD_dataset  -i  原数据集的路径  -overwrite_id 02```

3. nnUNetv2常用指令:

```bash
nnUNetv2_accumulate_crossval_results        nnUNetv2_find_best_configuration
nnUNetv2_apply_postprocessing               nnUNetv2_install_pretrained_model_from_zip
nnUNetv2_convert_MSD_dataset                nnUNetv2_move_plans_between_datasets
nnUNetv2_convert_old_nnUNet_dataset         nnUNetv2_plan_and_preprocess
nnUNetv2_determine_postprocessing           nnUNetv2_plan_experiment
nnUNetv2_download_pretrained_model_by_url   nnUNetv2_plot_overlay_pngs
nnUNetv2_ensemble                           nnUNetv2_predict
nnUNetv2_evaluate_folder                    nnUNetv2_predict_from_modelfolder
nnUNetv2_evaluate_simple                    nnUNetv2_preprocess
nnUNetv2_export_model_to_zip                nnUNetv2_train
nnUNetv2_extract_fingerprint
```

4. 分布式训练

- 设置显卡: ```export CUDA_VISIBLE_DEVICES=0,1```
- 分布式训练: ```nnUNetv2_train 220 2d 0 -num_gpus 2```

5. 预测

```nnUNetv2_predict -i ${nnUNet_raw}/Dataset131_WORD/ImagesTs -o output -d 131 -c 3d_fullres -f 1```

