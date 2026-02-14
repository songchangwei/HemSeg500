import nibabel as nib
import numpy as np
import os



def brain_to_mask(systhseg_file_path,filtered_file_path):
    # 加载数据
    systhseg_nii = nib.load(systhseg_file_path)

    systhseg_data = systhseg_nii.get_fdata()

    unique_values = np.unique(systhseg_data)
    print("独特的像素值有：", unique_values)

    # 要保留的像素值
    values_to_keep = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 26,
                          28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 58, 60]
    # 创建一个与原始数据大小相同的空数组，并初始化为0
    filtered_data = np.zeros_like(systhseg_data)
    # 将要保留的值进行赋值为1
    for value in values_to_keep:
        filtered_data[systhseg_data == value] = 1
    
    # 保存过滤后的数据到新的 NIfTI 文件
    filtered_nii = nib.Nifti1Image(filtered_data, systhseg_nii.affine, systhseg_nii.header)
    nib.save(filtered_nii, filtered_file_path)
    
def main():
    systhseg_folder = '/home/user512-001/songcw/code/naochuxue/SynthSeg/data/test_label_resample_2'
    filtered_floder = '/home/user512-001/songcw/code/naochuxue/SynthSeg/data/test_label_mask'
    for item in os.listdir(systhseg_folder):
        systhseg_file_path = os.path.join(systhseg_folder,item)
        filtered_file_path = os.path.join(filtered_floder,item)
        print(systhseg_file_path,filtered_file_path)
        brain_to_mask(systhseg_file_path,filtered_file_path)

if __name__=='__main__':
    main()