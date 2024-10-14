# Johof的lungmask

*Created by KennyS*

---

## 参考

[lungmask仓库](https://github.com/JoHof/lungmask/tree/master?tab=readme-ov-file#COVID-19-Web)

---


## 概览

1. 语义标签

    - Two label models (Left-Right):
        - 1 = Right lung
        - 2 = Left lung

    - Five label models (Lung lobes):
        - 1 = Left upper lobe
        - 2 = Left lower lobe
        - 3 = Right upper lobe
        - 4 = Right middle lobe
        - 5 = Right lower lobe

---


## 安装

`pip install lungmask`
`pip install git+https://github.com/JoHof/lungmask`

---


## 使用

### CLI

`lungmask INPUT OUTPUT`
`lungmask INPUT OUTPUT --modelname LTRCLobes`

### Python调用

1. 默认调用

    ```python
    from lungmask import LMInferer
    import SimpleITK as sitk

    inferer = LMInferer()

    input_image = sitk.ReadImage(INPUT)
    segmentation = inferer.apply(input_image)  # default model is U-net(R231)

    inferer = LMInferer(modelname="R231CovidWeb")
    inferer = LMInferer(modelname='LTRCLobes', fillmodel='R231')
    ```

2. 模型说明

    - R231：默认UNet模型
    - LTRCLobes：肺叶分割模型
    - R231CovidWeb：针对肺炎的肺分割模型

3. 自己修改并使用

    - `inferer.apply(input_image)`返回的是一个npy，需要转换为sitk_img并写入
    - 权重需要下载至`~/.cache/torch/hub/checkpoints`

    ```python
    inferer = LMInferer(modelname='R231CovidWeb', fillmodel='R231')

    for nifti_file in chosen_list:

        basename = os.path.basename(nifti_file)

        input_img = sitk.ReadImage(nifti_file)
        seg_array = inferer.apply(input_img)
        seg_img = sitk.GetImageFromArray(seg_array)

        output_dir = "/home/kennys/experiment/Vessels/processing/johof/R231CovidWeb_R231"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, basename)

        sitk.WriteImage(seg_img, output_path)
    ```