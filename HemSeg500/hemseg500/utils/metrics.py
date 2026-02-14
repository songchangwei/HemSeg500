from monai.metrics import DiceMetric,MeanIoU,HausdorffDistanceMetric,SurfaceDistanceMetric
from utils.eval_bootstrap_ci import cal_avg_bootstrap_confidence_interval
import os
import numpy as np
import pandas as pd
import nibabel as nib
from monai.transforms import LoadImage
import torch


def calculation_evaluation_metrics(result):
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric =  MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95, reduction="mean", get_not_nans=False)
    
    for item in result:
        dice_metric(y_pred=item['prediction'], y=item['label'])
        iou_metric(y_pred=item['prediction'], y=item['label'])
        hd_metric(y_pred=item['prediction'], y=item['label'])     
        
    
    print("evaluation dice metric:", dice_metric.aggregate().item())
    print("evaluation iou metric:", iou_metric.aggregate().item())
    print("evaluation hd metric:", hd_metric.aggregate().item())
    
    dice_metric.reset()
    iou_metric.reset()
    hd_metric.reset()
    
        


def calculation_evaluation_metrics_per_data(result):
    '''
    输入  result: [{"prediction":prediction,"label":label},{"prediction":prediction,"label":label}]
    输出  dice metric, iou metric, hd metric, sd metric
    '''
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric =  MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    hd_metric = HausdorffDistanceMetric(include_background=True,percentile=95, reduction="mean", get_not_nans=False)
    #sd_metric = SurfaceDistanceMetric(include_background=True,reduction="mean",get_not_nans=False)
    
    # 存储所有结果
    dice_scores = []
    iou_scores = []
    hd_scores = []

    
    
    for item in result:
        
        dice_metric(y_pred=item['prediction'], y=item['label'])
        iou_metric(y_pred=item['prediction'], y=item['label'])
        hd_metric(y_pred=item['prediction'], y=item['label'])

        # 获取度量结果
        dice_score = dice_metric.aggregate().item()
        iou_score = iou_metric.aggregate().item()
        hd_score = hd_metric.aggregate().item()

        # 将结果添加到列表中
        dice_scores.append(dice_score)
        iou_scores.append(iou_score)
        hd_scores.append(hd_score)

        # 重置度量以便进行下一次计算
        dice_metric.reset()
        iou_metric.reset()
        hd_metric.reset()
        
    # 转换列表为 NumPy 数组
    dice_scores_np = np.array(dice_scores)
    iou_scores_np = np.array(iou_scores)
    hd_scores_np = np.array(hd_scores)
        
    return dice_scores_np, iou_scores_np, hd_scores_np


def calculation_evaluation_metrics_and_ci(result):
    dice_scores, iou_scores, hd_scores = calculation_evaluation_metrics_per_data(result)
    print(iou_scores)
    dice_scores = np.nan_to_num(dice_scores, nan=0.0)
    iou_scores = np.nan_to_num(iou_scores, nan=0.0)
    hd_scores = np.nan_to_num(hd_scores, nan=0.0)
    dice_results = cal_avg_bootstrap_confidence_interval(dice_scores.reshape(-1))
    iou_results = cal_avg_bootstrap_confidence_interval(iou_scores.reshape(-1))
    hd_results = cal_avg_bootstrap_confidence_interval(hd_scores.reshape(-1))
    return dice_results,iou_results,hd_results,dice_scores

    

if __name__ == "__main__":
    
    # 预测的分割图像地址
    predicted_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/sam2_label/box_and_points/"
    # 真实的分割图像地址
    ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label/"

    # 使用 MONAI 的 LoadImage 来读取 nii.gz 文件
    load_image = LoadImage(image_only=True)

    # 定义CSV文件的路径
    csv_file_path = '/home/a41541/songcw/code/HemSeg500/tmp/test_set.csv'

    # 使用pandas read_csv方法和chunksize参数逐行读取文件
    chunksize = 1  # 每次读取一行
    result = []
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize):
        # chunk是一个DataFrame，仅包含一行数据
        filename = chunk['filename'].values[0]  # 获取当前行的filename
        label = chunk['label'].values[0]        # 获取当前行的label
        predicted_segmentation_file_path = predicted_segmentation_path + filename + '.nii.gz'
        ground_truth_segmentation_file_path = ground_truth_segmentation_path + filename + '.nii.gz'
        predicted_segmentation = load_image(predicted_segmentation_file_path)
        ground_truth_segmentation = load_image(ground_truth_segmentation_file_path)
        #   转换为 PyTorch 张量，并添加 batch 维度和 channel 维度
        predicted = predicted_segmentation.clone().detach().unsqueeze(0).unsqueeze(0).float()
        ground_truth = ground_truth_segmentation.clone().detach().unsqueeze(0).unsqueeze(0).float()
        result.append({'prediction':predicted,'label':ground_truth})
        
    dice_scores, iou_scores, hd_scores = calculation_evaluation_metrics_per_data(result)
    #print(dice_scores.reshape(-1))
    #print(iou_scores.reshape(-1))
    #print(hd_scores.reshape(-1))
    dice_scores = np.nan_to_num(dice_scores, nan=0.0)
    iou_scores = np.nan_to_num(iou_scores, nan=0.0)
    hd_scores = np.nan_to_num(hd_scores, nan=0.0)
    dice_results = cal_avg_bootstrap_confidence_interval(dice_scores.reshape(-1))
    iou_results = cal_avg_bootstrap_confidence_interval(iou_scores.reshape(-1))
    hd_results = cal_avg_bootstrap_confidence_interval(hd_scores.reshape(-1))
    print('dice_scores:',dice_results)
    print('iou_scores:',iou_results)
    print('hd_scores:',hd_results)



