import logging
import os,random
import sys
import tempfile
from glob import glob
import nibabel as nib
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import ImageDataset, create_test_image_3d, decollate_batch, DataLoader, pad_list_data_collate,list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    RandRotate90,
    RandSpatialCrop,
    ScaleIntensity,
    Spacingd,
    Orientationd,
    RandCropByPosNegLabel,
    LoadImaged,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
)
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import Transform,MapTransform
from torch.optim.lr_scheduler import StepLR
from monai.networks.nets import UNet


def make_deterministic(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
make_deterministic(42)


class CTNormalizationd(MapTransform):
    def __init__(self, keys, intensity_properties, target_dtype=np.float32):
        """
        初始化CTNormalization转换。
        :param keys: 字典中要转换的键列表
        :param intensity_properties: 包含强度相关属性的字典（均值、标准差、百分位数边界等）
        :param target_dtype: 转换目标的数据类型
        """
        super().__init__(keys)
        self.intensity_properties = intensity_properties
        self.target_dtype = target_dtype

    def __call__(self, data):
        """
        在图像上应用CT标准化。
        :param data: 包含图像数据的字典
        :return: 包含标准化图像数据的字典
        """
        d = dict(data)
        for key in self.keys:
            assert self.intensity_properties is not None, "CTNormalizationd requires intensity properties"
            d[key] = d[key].astype(self.target_dtype)
            mean_intensity = self.intensity_properties['mean']
            std_intensity = self.intensity_properties['std']
            lower_bound = self.intensity_properties['percentile_00_5']
            upper_bound = self.intensity_properties['percentile_99_5']
            d[key] = np.clip(d[key], lower_bound, upper_bound)
            d[key] = (d[key] - mean_intensity) / max(std_intensity, 1e-8)
        return d


def main(tempdir):

    images = sorted(glob(os.path.join(tempdir, 'train/image',"*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, 'train/label',"*.nii.gz")))

    val_images = sorted(glob(os.path.join(tempdir, 'val/image',"*.nii.gz")))
    val_segs = sorted(glob(os.path.join(tempdir, 'val/label',"*.nii.gz")))
    
    train_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]
    val_files = [{"image": img, "label": seg} for img, seg in zip(val_images, val_segs)]
    
    train_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.4882810115814209,0.4882810115814209,4.649400138854981),
            mode=("bilinear", "nearest"),
        ),
        CTNormalizationd(keys=['image'],intensity_properties={'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),
        RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(256,256,16), pos=1,neg=1,num_samples=4,image_key='image',image_threshold=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ]
    )
    val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.4882810115814209,0.4882810115814209,4.649400138854981),
            mode=("bilinear", "nearest"),
        ),
        CTNormalizationd(keys=['image'],intensity_properties={'mean':48.13441467285156,'std':13.457549095153809,'percentile_00_5':11.99969482421875,'percentile_99_5':79.0}),
        #RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(256,256,16), pos=1,neg=1,num_samples=4,image_key='image',image_threshold=0,),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ]
    )
    
    
    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transform)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    
    train_check_data = monai.utils.misc.first(train_loader)
    print(train_check_data["image"].shape, train_check_data["label"].shape)
    
    val_check_data = monai.utils.misc.first(val_loader)
    print(val_check_data["image"].shape, val_check_data["label"].shape)
    
    
if __name__ == "__main__":
    temdir = '/home/user512-001/songcw/data/naochuxue'
    main(temdir)