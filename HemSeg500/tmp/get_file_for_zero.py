import os
import nibabel as nib
import numpy as np

def count_foreground_background(file_path, threshold=0.5):
    # 读取 NIfTI 文件
    img = nib.load(file_path)
    data = img.get_fdata()
    
    # 二值化
    foreground = data > threshold
    background = data <= threshold
    
    # 计算前景和背景的样本数
    num_foreground = np.count_nonzero(foreground)
    num_background = np.count_nonzero(background)
    
    return num_foreground, num_background

def find_files_with_zero_foreground(folder_path, threshold=0.5):
    # 获取文件夹中的所有 NIfTI 文件
    nii_files = [f for f in os.listdir(folder_path) if f.endswith('.nii.gz')]
    
    zero_foreground_files = []
    
    for nii_file in nii_files:
        file_path = os.path.join(folder_path, nii_file)
        num_foreground, num_background = count_foreground_background(file_path, threshold)
        
        if num_foreground == 0:
            zero_foreground_files.append(nii_file)
        
        print(f"File: {nii_file}")
        print(f"Num foregrounds: {num_foreground}")
        print(f"Num backgrounds: {num_background}\n")
    
    return zero_foreground_files

# 调用函数，传入文件夹路径
folder_path = '/home/a4154/songcw/data/naochuxue/manual_label'
# 找到前景为 0 的文件
zero_foreground_files = find_files_with_zero_foreground(folder_path)

# 输出前景为 0 的文件
print("Files with zero foreground elements:")
for file in zero_foreground_files:
    print(file)
