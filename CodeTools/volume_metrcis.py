# 计算三维下各种指标
from __future__ import absolute_import, print_function

import pandas as pd
import GeodisTK
import numpy as np
from scipy import ndimage


# pixel accuracy
def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s == g).sum()) / g.size
    return pa


# Dice evaluation
def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return dice


# IOU evaluation
def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if (image_dim == 2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim == 3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """
    tp = np.sum((pred == 1) & (label == 1))
    tn = np.sum((pred == 0) & (label == 0))
    fp = np.sum((pred == 1) & (label == 0))
    fn = np.sum((pred == 0) & (label == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if (len(s_volume.shape) == 4):
        assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0] == 1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower == "dice"):
        score = binary_dice(s_volume, g_volume)

    elif (metric_lower == "iou"):
        score = binary_iou(s_volume, g_volume)

    elif (metric_lower == 'assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif (metric_lower == "hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif (metric_lower == "rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower == "volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score

# 新增处理多类别的评估计算
def multi_class_metrics(seg_arr, gd_arr):
    num_classes = np.unique(gd_arr)
    print(f"Num classes is {num_classes}")

    results = {}
    
    # dice_sum = 0
    # iou_sum = 0
    # # hd_sum = 0
    # rve_sum = 0
    # sens_sum = 0
    # spec_sum = 0

    for c in num_classes:
        seg_bin = (seg_arr == c).astype(np.float32)
        gd_bin = (gd_arr == c).astype(np.float32)

        dice = binary_dice(seg_bin, gd_bin)
        iou = binary_iou(seg_bin, gd_bin)
        # hd = binary_hausdorff95(seg_bin, gd_bin)
        rve = binary_relative_volume_error(seg_bin, gd_bin)

        sens, spec = compute_class_sens_spec(seg_bin, gd_bin)

        # dice_sum += dice
        # iou_sum += iou
        # # hd_sum += hd
        # rve_sum += rve
        # sens_sum += sens
        # spec_sum += spec

        results[c] = {
            "Dice": dice,
            "IOU": iou,
            "RVE": rve,
            "Sen": sens,
            "Spec": spec,
            # "HD95": hd
        }

    # # return dices, ious, rves, senss, specs # hds
    # results['average'] = {
    #     "Dice": dice_sum / len(num_classes),
    #     "IOU": iou_sum / len(num_classes),
    #     "RVE": rve_sum / len(num_classes),
    #     "Sen": sens_sum / len(num_classes),
    #     "Spec": spec_sum / len(num_classes),
    #     "HD95": hd_sum / len(num_classes)
    # }
    return results

"""
seg_path = nnUNetTrainer_dir
gd_path = labelTs_dir
save_dir = "/home/kennys/experiment/MultiOrgan/csv"
seg = sorted(os.listdir(gd_path))
case_name = []

nnUNetTrainer_perf_metrics = []

for name in seg:
    if not name.startswith('.') and name.endswith('nii.gz'):
        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(seg_path, name)))
        gd_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gd_path, name)))
        case_name.append(name)

        results = multi_class_metrics(seg_arr, gd_arr)

        for class_id, metrics in results.items():
            metrics["ClassID"] = class_id
            metrics["SeriesUID"] = name.split(".nii.gz")[0]
            nnUNetTrainer_perf_metrics.append(metrics)

nnUNetTrainer_perf_metrics_df = pd.DataFrame(nnUNetTrainer_perf_metrics)

nnUNetTrainer_perf_metrics_df.to_csv(os.path.join(save_dir, 'nnUNetTrainer_perf.csv'))
"""