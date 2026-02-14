import os
import nibabel as nib
import numpy as np
import torch
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from utils.eval_bootstrap_ci import cal_avg_bootstrap_confidence_interval
import pandas as pd

def load_nifti(file_path):
    """Load a NIfTI file and return a numpy array."""
    image = nib.load(file_path)
    return image.get_fdata()

def preprocess_image(image):
    """Preprocess the image: binarize and convert to tensor."""
    # Assuming the images are binary masks
    image = (image > 0).astype(np.float32)
    return torch.tensor(image)

def evaluate_segmentation(predicted_path, ground_truth_path):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")

    results = {"filename": [], "dice": [], "iou": [], "hd95": []}

    # Iterate over files in the directories
    dice_list = []
    for filename in os.listdir(predicted_path):
        pred_file = os.path.join(predicted_path, filename)
        gt_file = os.path.join(ground_truth_path, filename)

        if os.path.exists(pred_file) and os.path.exists(gt_file):
            pred_image = load_nifti(pred_file)
            gt_image = load_nifti(gt_file)

            pred_tensor = preprocess_image(pred_image).unsqueeze(0).unsqueeze(0)
            gt_tensor = preprocess_image(gt_image).unsqueeze(0).unsqueeze(0)

            # Calculate metrics
            dice = dice_metric(pred_tensor, gt_tensor).item()
            iou = iou_metric(pred_tensor, gt_tensor).item()
            hd95 = hd95_metric(pred_tensor, gt_tensor).item()
            
            dice_list.append(dice)

            # Store the results
            results["filename"].append(filename)
            results["dice"].append(dice)
            results["iou"].append(iou)
            results["hd95"].append(hd95)

            print(f"File: {filename}")
            print(f"  Dice Coefficient: {dice}")
            print(f"  IoU: {iou}")
            print(f"  Hausdorff Distance (95th percentile): {hd95}")

    # Save results to a .npy file
    print(results)
    dice_result = cal_avg_bootstrap_confidence_interval(results['dice'])
    iou_result = cal_avg_bootstrap_confidence_interval(results['iou'])
    hd_result = cal_avg_bootstrap_confidence_interval(results['hd95'])
    print('dice_result',dice_result)
    print('iou_result',iou_result)
    print('hd_result',hd_result)
    #np.save(output_path, results)
    #print(f"Results saved to {output_path}")
    return dice_list

if __name__ == "__main__":
    
    # 预测的分割图像地址
    #predicted_segmentation_path = "/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/points/points_9"
    predicted_segmentation_path = "/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/box_and_points/points_4"
    #predicted_segmentation_path = "/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/boxs"
    # 真实的分割图像地址
    ground_truth_segmentation_path = "/home/user512-003/songcw/data/naochuxue/test/label/test_label"
    
    #predicted_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/sam2_label/points_ivh_iph/iph"
    #predicted_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/sam2_label/points_ivh_iph/ivh"
    #ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label_ivh_iph/iph"
    #ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label_ivh_iph/ivh"
    
    dice_scores = evaluate_segmentation(predicted_segmentation_path, ground_truth_segmentation_path)
    # 将列表转换为 DataFrame（作为一列）
    df = pd.DataFrame(dice_scores, columns=['points'])

    # 保存为 CSV 文件
    df.to_csv('dice_result/sam2_box_and_points_4_dice.csv', index=False)
