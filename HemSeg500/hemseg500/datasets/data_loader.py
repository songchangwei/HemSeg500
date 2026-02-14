import os,random
from glob import glob
import torch
import monai
from monai.data import  DataLoader, list_data_collate
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    Spacingd,
    Orientationd,
    LoadImaged,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    Activations,
    AsDiscrete,
    ScaleIntensityd,
    RandRotate90d,
    SpatialPadd,
)
import numpy as np
import sys
sys.path.append('/home/a4154/songcw/code/HemSeg500/hemseg500')
from utils import CTNormalizationd



def train_trans():
    train_transform = Compose(
        [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(64,64,32)),  # Ensure minimum size of 32 in each dimension
        RandCropByPosNegLabeld(keys=['image', 'label'],label_key='label',spatial_size =(64,64,32), pos=2,neg=2,num_samples=4,image_key='image',image_threshold=0),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ]
        )
    return train_transform

    
def test_trans():
    test_transform = Compose(
    [
        LoadImaged(keys=["prediction", "label"]),
        EnsureChannelFirstd(keys=["prediction", "label"]),
        EnsureTyped(keys=["prediction", "label"]),
        Orientationd(keys=["prediction", "label"], axcodes="RAS"),
    ]
    )
    return test_transform


def val_trans():
    val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys="image"),
    ]
    )
    return val_transform

def data_preprocessing_for_test(predicted_segmentation_path, ground_truth_segmentation_path):
    prediction = sorted(glob(os.path.join(predicted_segmentation_path, "*.nii.gz")))
    ground_truth = sorted(glob(os.path.join(ground_truth_segmentation_path, "*.nii.gz")))
    test_files = [{"prediction": img, "label": seg} for img, seg in zip(prediction, ground_truth)]
    
    test_ds = monai.data.Dataset(data=test_files, transform=test_trans())
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)

    return test_loader



def data_preprocessing_for_inference(tempdir):
    images = sorted(glob(os.path.join(tempdir, 'test/image',"*.nii.gz")))
    segs = sorted(glob(os.path.join(tempdir, 'test/label',"*.nii.gz")))
    val_files = [{"image": img, "label": seg, "file_name":img.split('/')[-1].split('.')[0]} for img, seg in zip(images, segs)]
    
    val_ds = monai.data.Dataset(data=val_files, transform=val_trans())
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    #val_check_data = monai.utils.misc.first(val_loader)
    #print(val_check_data["image"].shape, val_check_data["label"].shape)
    return val_loader

def data_preprocessing_for_train(tempdir,batch_size):

    train_images = sorted(glob(os.path.join(tempdir, 'train/image',"*.nii.gz")))
    train_segs = sorted(glob(os.path.join(tempdir, 'train/label',"*.nii.gz")))

    val_images = sorted(glob(os.path.join(tempdir, 'val/image',"*.nii.gz")))
    val_segs = sorted(glob(os.path.join(tempdir, 'val/label',"*.nii.gz")))
    
    train_files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_segs)]
    val_files = [{"image": img, "label": seg} for img, seg in zip(val_images, val_segs)]
    
    
    # Prepare dataset
    train_ds = monai.data.Dataset(data=train_files, transform=train_trans())

  
    # Create DataLoader with WeightedRandomSampler
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=1,
        collate_fn=monai.data.list_data_collate,
        pin_memory=torch.cuda.is_available()
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_trans())
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, collate_fn=list_data_collate)
    

    
    return train_loader,val_loader
    

def post_trans():
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    return post_trans

if __name__ == "__main__":
    train_loader,val_loader = data_preprocessing_for_train('/home/a4154/songcw/data/naochuxue',8)
    train_check_data = monai.utils.misc.first(train_loader)
    print(f'train_check_image:{train_check_data["image"].shape},     train_check_label:{train_check_data["label"].shape}')

