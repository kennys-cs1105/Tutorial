import os

import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from medicalimaging import CTtools


import numpy as np


def ct_seg_perf(
    pred_mask=None,
    true_mask=None,
    binarize=False,
    pred_mask_path=None,
    true_mask_path=None,
):
    """
    CT图像分割性能评估。

    参数:
    pred_mask (np.ndarray): 预测的分割掩码。
    true_mask (np.ndarray): 真实的分割掩码。
    binarize (bool): 是否将掩码二值化，默认为False。
    pred_path (str): 预测掩码的文件路径。
    true_path (str): 真实掩码的文件路径。

    返回:
    dict: 包含每个类别的Dice系数、IOU分数、准确率、精确率、召回率、F1分数和假阳性率的字典。
    """
    # 如果提供了文件路径，则从文件中读取掩码
    if pred_mask_path is not None:
        pred_mask = read_mask(pred_mask_path)
    if true_mask_path is not None:
        true_mask = read_mask(true_mask_path)

    if binarize:
        pred_mask = (pred_mask > 0.5).astype(np.int8)
        true_mask = (true_mask > 0.5).astype(np.int8)

    # 获取所有前景类别
    classes = np.unique(true_mask)
    classes = classes[classes != 0]  # 移除背景类别

    # 初始化结果字典
    results = {}

    # 对每个类别计算性能指标
    for class_id in classes:
        pred_binary_mask = (pred_mask == class_id).astype(np.int8)
        true_binary_mask = (true_mask == class_id).astype(np.int8)

        # 计算交集和并集
        intersection = np.bitwise_and(pred_binary_mask, true_binary_mask)
        union = np.bitwise_or(pred_binary_mask, true_binary_mask)
        false_positive = np.bitwise_and(
            pred_binary_mask, np.bitwise_not(true_binary_mask)
        )

        # 计算性能指标
        dice = (
            2 * intersection.sum() / (pred_binary_mask.sum() + true_binary_mask.sum())
            if pred_binary_mask.sum() + true_binary_mask.sum() > 0
            else 0
        )
        iou = intersection.sum() / union.sum() if union.sum() > 0 else 0
        accuracy = (
            np.equal(pred_binary_mask, true_binary_mask).sum() / true_binary_mask.size
        )
        precision = (
            intersection.sum() / pred_binary_mask.sum()
            if pred_binary_mask.sum() > 0
            else 0
        )
        recall = (
            intersection.sum() / true_binary_mask.sum()
            if true_binary_mask.sum() > 0
            else 0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        fp_rate = (
            false_positive.sum() / (false_positive.sum() + true_binary_mask.sum())
            if (false_positive.sum() + true_binary_mask.sum()) > 0
            else 0
        )

        # 将结果添加到字典
        results[class_id] = {
            "Dice": dice,
            "IOU": iou,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FPrate": fp_rate,
        }

    return results


def read_mask(mask_path):
    """
    从给定的文件路径读取掩码。

    参数:
    mask_path (str): 掩码的文件路径。

    返回:
    np.ndarray: 读取的掩码数组。
    """
    if mask_path.endswith(".npy"):
        return np.load(mask_path)
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(mask_path))


# 示例用法
# result = ct_seg_perf(pred_path='path_to_pred_mask.nii', true_path='path_to_true_mask.nii')


def get_array(source):
    """
    根据提供的源数据类型，返回相应的NumPy数组。

    参数:
    source: 可以是文件路径字符串、SimpleITK.Image对象或NumPy数组。

    返回:
    NumPy数组: 从给定的源数据中提取或直接返回的NumPy数组。

    如果源数据是字符串，假定它是掩码文件的路径，并调用read_mask函数来读取。
    如果源数据是SimpleITK.Image对象，使用SimpleITK将其转换为NumPy数组。
    如果源数据已经是NumPy数组，则直接返回。
    如果源数据类型不支持，将抛出ValueError。
    """
    if isinstance(source, str):
        # 假定source是文件路径，调用read_mask函数读取掩码文件
        return read_mask(source)
    elif isinstance(source, sitk.Image):
        # 如果source是SimpleITK.Image对象，转换为NumPy数组
        return sitk.GetArrayFromImage(source)
    elif isinstance(source, np.ndarray):
        # 如果source已经是NumPy数组，直接返回
        return source
    else:
        # 如果source不是上述支持的类型，抛出异常
        raise ValueError("Unsupported source type.")


def evaluate_lesion_inclusion(
    ct_source=None,
    organ_source=None,
    lesion_source=None,
    binarize=True,
    display=False,
    inclusion_threshold=0.95,
    save_images=False,
    save_dir=None,
    ct_id=None,
    preset="lung",
):
    """
    评估结节是否位于指定器官掩码内，并可选地在CT图像上绘制结节。

    参数:
    ct_source: 可以是CT图像的文件路径、SimpleITK.Image对象或NumPy数组。
    organ_source: 可以是器官掩码的文件路径、SimpleITK.Image对象或NumPy数组。
    lesion_source: 可以是结节掩码的文件路径、SimpleITK.Image对象或NumPy数组。
    binarize: 是否将掩码二值化，默认为True。
    display: 是否在屏幕上显示结节绘制结果，默认为False。
    inclusion_threshold: 结节与器官掩码交集的阈值，用于判断结节是否被包含，默认为0.95。
    save_images: 是否保存绘制的结节图像，默认为False。
    save_dir: 保存图像的目录路径，仅当save_images为True时需要。
    ct_id: CT的编号，用于命名保存的图像，仅当save_images为True时需要。

    返回:
    inclusion_ratio: 结节被器官掩码包含的比例，值域为[0, 1]。
    is_included: 布尔值，表示结节是否被器官掩码包含。

    该函数首先检查传入的源参数类型，然后读取或转换为NumPy数组。
    接着，如果需要，会对掩码进行二值化处理。
    然后，计算结节掩码与器官掩码的交集，并根据交集面积评估结节是否被包含。
    如果设置为显示或保存图像，函数还会在CT图像上绘制结节并进行相应的操作。
    """

    if save_images and not all([ct_id, save_dir]):
        missing_params = ", ".join(
            [
                param
                for param, value in {"ct_id": ct_id, "save_dir": save_dir}.items()
                if not value
            ]
        )
        raise ValueError(f"必须指定以下参数：{missing_params}")

    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    ct_array = get_array(ct_source) if ct_source is not None else None
    # 检查数据类型，如果不是uint8，则应用窗宽窗位调整
    if ct_array.dtype != np.uint8:
        ct_array = CTtools.apply_windowing(ct_array, preset=preset, apply_limits=False)
    organ_mask = get_array(organ_source) if organ_source is not None else None
    lesion_mask = get_array(lesion_source) if lesion_source is not None else None

    # 二值化处理
    if binarize:
        organ_mask = (organ_mask > 0).astype(np.int8)
        lesion_mask = (lesion_mask > 0).astype(np.int8)
    else:
        raise ValueError(
            "Binarize parameter is set to False, but multi-class processing is not implemented."
        )

    total_intersection = 0
    total_lesion_area = 0

    # 遍历每个横断面
    for slice_index in range(organ_mask.shape[0]):
        lesion_slice = lesion_mask[slice_index, :, :]
        if not lesion_slice.any():
            continue
        lesion_area = np.sum(lesion_slice)
        mask_slice = organ_mask[slice_index, :, :]
        intersection_area = np.sum(np.logical_and(mask_slice, lesion_slice))

        total_intersection += intersection_area
        total_lesion_area += lesion_area

        # 显示或保存图像
        if (
            display
            or save_images
            or (intersection_area / lesion_area < inclusion_threshold)
        ):
            slice_img = ct_array[slice_index, :, :].copy()
            slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
            lesion_slice = lesion_slice.astype(np.uint8)
            contours, _ = cv2.findContours(
                lesion_slice,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(slice_img, contours, -1, (0, 0, 255), 2)

            if mask_slice.any():
                mask_slice = mask_slice.astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(slice_img, contours, -1, (0, 255, 255), 2)

            cv2.putText(
                slice_img,
                f"{intersection_area / lesion_area:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            if save_images:
                cv2.imwrite(
                    os.path.join(save_dir, f"{ct_id}.{slice_index}.png"), slice_img
                )
            if display or (intersection_area / lesion_area < inclusion_threshold):
                plt.imshow(slice_img[:, :, ::-1])
                plt.show()

    # 计算包含比例
    inclusion_ratio = (
        total_intersection / total_lesion_area if total_lesion_area > 0 else 0
    )
    is_included = inclusion_ratio >= inclusion_threshold

    return inclusion_ratio, is_included


def lesion_recall(
    region_source=None,
    lesion_source=None,
):
    """
    计算病灶的召回率。

    参数:
    region_source: 可以是预测的包含病灶的区域掩码的文件路径、SimpleITK.Image对象或NumPy数组。
    lesion_source: 可以是真实的病灶掩码的文件路径、SimpleITK.Image对象或NumPy数组。

    返回:
    recall: 计算得到的病灶召回率。
    """

    # 读取或转换为NumPy数组
    eval_region_array = get_array(region_source)
    lesion_mask_array = get_array(lesion_source)

    # 对掩码进行二值化处理

    eval_region_array = (eval_region_array > 0.5).astype(np.int8)
    lesion_mask_array = (lesion_mask_array > 0.5).astype(np.int8)

    # 计算召回率
    true_positives = np.bitwise_and(eval_region_array, lesion_mask_array).sum()
    possible_positives = lesion_mask_array.sum()
    recall = true_positives / possible_positives if possible_positives > 0 else 0

    return recall
