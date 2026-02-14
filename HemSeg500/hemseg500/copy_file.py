import os
import shutil

def add_suffix_and_copy_recursive(source_folder, dest_folder, suffix='.nii.gz'):
    """
    遍历source_folder及其所有子文件夹中的文件，为每个文件名添加指定后缀，并复制到dest_folder。

    参数：
    - source_folder (str): 源文件夹路径。
    - dest_folder (str): 目标文件夹路径。
    - suffix (str): 要添加到每个文件名末尾的后缀，默认为'.gz.nii'。
    """
    # 检查并创建目标文件夹
    os.makedirs(dest_folder, exist_ok=True)

    # 使用os.walk递归遍历源文件夹
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            # 构建源文件的完整路径
            source_path = os.path.join(root, filename)
            
            # 确认源路径确实是文件
            if os.path.isfile(source_path):
                # 添加后缀到文件名
                new_filename = filename + suffix
                target_path = os.path.join(dest_folder, new_filename)
                
                # 检查目标文件是否已存在，避免覆盖
                if os.path.exists(target_path):
                    print(f"警告: 目标文件 {new_filename} 已存在，跳过复制。")
                    continue
                
                # 复制文件到目标文件夹
                shutil.copy(source_path, target_path)
                print(f"已复制: {source_path} -> {target_path}")

if __name__ == "__main__":
    source_folder = '/home/user512-001/songcw/data/naochuxue/test/sam2_label/box_and_points_ivh_iph'  # 源文件夹路径
    dest_folder = '/home/user512-001/songcw/data/naochuxue/test/sam2_label/box_and_points'  # 目标文件夹路径

    # 调用函数
    add_suffix_and_copy_recursive(source_folder, dest_folder)
