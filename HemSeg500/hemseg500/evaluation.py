from utils import calculation_evaluation_metrics_and_ci,calculation_evaluation_metrics
from datasets import data_preprocessing_for_test
import os
import pandas as pd

if __name__ == "__main__":
    
    # 预测的分割图像地址
    predicted_segmentation_path = "/home/user512-001/songcw/code/naochuxue/HemSeg500/result/unet3D"
    # 真实的分割图像地址
    ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label/test_label"
    
    #IPH
    #predicted_segmentation_path = "/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/prediction_3d_nii_ivh_iph/iph"
    #ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label_ivh_iph/iph"
    
    #IVH
    predicted_segmentation_path = "/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/prediction_3d_nii_ivh_iph/ivh"
    ground_truth_segmentation_path = "/home/user512-001/songcw/data/naochuxue/test/label_ivh_iph/ivh"
    
    test_loader = data_preprocessing_for_test(predicted_segmentation_path, ground_truth_segmentation_path)
    result = []
    
    for _, test_data in enumerate(test_loader, start=1):
        predictions, labels = test_data['prediction'], test_data['label']
        result.append({'prediction':predictions,'label':labels})
        print(predictions.shape,labels.shape)
    dice_results,iou_results,hd_results,dice_scores = calculation_evaluation_metrics_and_ci(result)
    print('dice_results:',dice_results)
    print('iou_results:',iou_results)
    print('hd_results:',hd_results)
    
    print(dice_scores)
    
    # 将列表转换为 DataFrame（作为一列）
    #df = pd.DataFrame(dice_scores, columns=['unet3d'])

    # 保存为 CSV 文件
    #df.to_csv('dice_result/unet3d_dice.csv', index=False)
        