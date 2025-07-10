import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from monai.data import Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    CropForegroundd, Orientationd, Spacingd, RandCropByPosNegLabeld,
    AsDiscrete, AsDiscreted, Invertd, SaveImaged
)
from monai.utils import set_determinism
from monai.networks.nets import UNet, SegResNet, DynUNet, UNETR, SwinUNETR
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.config import print_config

print_config()


"""
MONAI 自定义训练测试
"""


# Configuration Constants
MODEL_TYPE = "SegResNet"
MAX_EPOCHS = 200
VAL_INTERVAL = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
SPATIAL_SIZE = ROI_SIZE = (96, 96, 96) # (160,160,160)
SW_BATCH_SIZE = 4
DEVICE = torch.device("cuda:0")
DATA_DIR = "..."
MODEL_PATH = "best_metric_model.pth"
OUTPUT_DIR = "..."


MODEL_CONFIGS = {
    "UNet": {"channels": (16, 32, 64, 128, 256), "strides": (2, 2, 2, 2), "num_res_units": 2, "norm": Norm.BATCH},
    "SegResNet": {"init_filters": 24, "blocks_down": [1, 2, 2, 4], "blocks_up": [1, 1, 1], "dropout_prob": 0.2},
    "DynUNet": {
        "kernel_size": [3] * 6,
        "strides": [1, 2, 2, 2, 2, [2, 2, 1]],
        "upsample_kernel_size": [2, 2, 2, 2, [2, 2, 1]],
        "norm_name": "instance",
        "deep_supervision": False,
        "res_block": True,
    },
    "UNETR": {
        "img_size": SPATIAL_SIZE, "feature_size": 64, "hidden_size": 1536,
        "mlp_dim": 3072, "num_heads": 48, "proj_type": "conv",
        "norm_name": "instance", "res_block": True, "dropout_rate": 0.1
    },
    "SwinUNETR": {"img_size": SPATIAL_SIZE, "feature_size": 48, "use_checkpoint": True},
}


def get_model(model_type, config):
    model_class_map = {
        "UNet": UNet,
        "SegResNet": SegResNet,
        "DynUNet": DynUNet,
        "UNETR": UNETR,
        "SwinUNETR": SwinUNETR
    }
    model = model_class_map[model_type](spatial_dims=3, in_channels=1, out_channels=2, **config)
    return model.to(DEVICE)


def get_transforms(is_train=True):
    # 自定义transforms
    # CropForegroundd功能慎用，它直接把你CT的空白部分去掉了，造成CT的size变化。如果你想让分割的结果再重叠到原CT，这个功能就禁用了
    # 注意train_transforms 和val_transforms 的超参数要一致
    # ScaleIntensityRanged结合实际任务的CT值范围
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys="image", a_min=-900, a_max=-300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
    if is_train:
        base.append(RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=SPATIAL_SIZE,
            pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0
        ))
    return Compose(base)


def get_test_transforms():
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-900, a_max=-300,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
    ])


def get_post_transforms():
    return Compose([
        Invertd(
            keys="pred",
            transform=get_test_transforms(),
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=OUTPUT_DIR,
            output_postfix="seg",
            resample=False
        ),
    ])


def prepare_data():
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
    label_paths = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]
    return data_dicts[:200], data_dicts[200:]


def get_test_loader(data_dir):
    test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
    test_data = [{"image": img} for img in test_images]
    test_ds = Dataset(data=test_data, transform=get_test_transforms())
    return DataLoader(test_ds, batch_size=1, num_workers=4)


def train():
    set_determinism(seed=0)
    train_files, val_files = prepare_data()
    
    # CacheDataset: 预加载所有原始数据。将non-random-transform应用到数据并提前缓存，提高加载速度。如果数据集数量不是特别大，能够全部缓存到内存中，这是性能最高的Dataset
    # dataset = CacheDataset(data=train_dict, trainsforms, cache_rate=1.0,num_workers=4, progress=True)
    # ---------------------------------------------------------------------------------------------------#
    # PersistentDataset: 用于处理大型数据集。将数据应用non-random-transform，缓存在硬盘中
    # dataset = PersistentDataset(data=train_dict, transforms, cache_dir="./data/cache")
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Dataset(train_files, get_transforms(True)), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(Dataset(val_files, get_transforms(False)), batch_size=1, num_workers=4)

    model = get_model(MODEL_TYPE, MODEL_CONFIGS[MODEL_TYPE])
    print(f"Using {MODEL_TYPE}, Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = Compose([AsDiscrete(argmax=True, to_onehot=2)]), Compose([AsDiscrete(to_onehot=2)])

    best_metric, best_epoch = -1, -1
    epoch_losses, val_metrics = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step + 1}/{len(train_loader)} train_loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_labels = val_batch["image"].to(DEVICE), val_batch["label"].to(DEVICE)
                    val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                val_metrics.append(metric)

                if metric > best_metric:
                    best_metric, best_epoch = metric, epoch + 1
                    save_path = os.path.join(DATA_DIR, f"best_metric_model_{MODEL_TYPE.lower()}.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved new best model to {save_path}")

                print(f"Epoch {epoch + 1} val dice: {metric:.4f}, best: {best_metric:.4f} (epoch {best_epoch})")

    # Plot metrics
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Epoch Loss - {MODEL_TYPE}")
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses)
    plt.xlabel("Epoch")
    plt.subplot(1, 2, 2)
    plt.title(f"Val Dice - {MODEL_TYPE}")
    plt.plot([VAL_INTERVAL * (i+1) for i in range(len(val_metrics))], val_metrics)
    plt.xlabel("Epoch")
    plt.savefig(os.path.join(DATA_DIR, f"training_metrics_{MODEL_TYPE.lower()}.png"), dpi=300)
    print(f"Training complete. Best metric: {best_metric:.4f} at epoch {best_epoch}")


def test():
    loader = get_test_loader(DATA_DIR)
    model = get_model(MODEL_TYPE, MODEL_CONFIGS)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    post_transforms = get_post_transforms()

    with torch.no_grad():
        for test_batch in loader:
            test_inputs = test_batch["image"].to(DEVICE)
            test_batch["pred"] = sliding_window_inference(
                test_inputs, ROI_SIZE, SW_BATCH_SIZE, model
            )
            outputs = [post_transforms(i) for i in decollate_batch(test_batch)]


if __name__ == "__main__":
    train()
    # test()