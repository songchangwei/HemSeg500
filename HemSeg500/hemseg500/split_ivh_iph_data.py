import pandas as pd
import os
import shutil

def copy_files_based_on_label(csv_file, source_folder, dest_folder_0, dest_folder_1):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 创建目标文件夹，如果不存在
    os.makedirs(dest_folder_0, exist_ok=True)
    os.makedirs(dest_folder_1, exist_ok=True)

    # 遍历每一行，将文件复制到对应的文件夹
    for index, row in df.iterrows():
        filename = row['filename']
        label = row['label']

        # 构建源文件路径
        source_path = os.path.join(source_folder, filename+'.nii.gz')
        if not os.path.exists(source_path):
            print(f"Warning: File {filename} does not exist in the source folder.")
            continue

        # 确定目标文件夹
        if label == 0:
            target_folder = dest_folder_0
        else:
            target_folder = dest_folder_1

        # 构造目标路径
        target_path = os.path.join(target_folder, filename+'.nii.gz')

        # 复制文件到目标文件夹
        print(source_path, target_path)
        shutil.copy(source_path, target_path)
        print(f"Copied {filename} to {target_folder}")

if __name__ == "__main__":
    # 示例调用
    csv_file = 'tmp/test_set.csv'          # CSV文件路径
    source_directory = '/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/prediction_3d_nii'         # 源文件夹路径
    target_directory_0 = '/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/prediction_3d_nii_ivh_iph/iph'     # 标签为0的目标文件夹
    target_directory_1 = '/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/prediction_3d_nii_ivh_iph/ivh'     # 标签为1的目标文件夹

    copy_files_based_on_label(csv_file, source_directory, target_directory_0, target_directory_1)
