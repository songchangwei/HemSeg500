import os
import shutil

def copy_unique_files(src_folder, dst_folder):
    # 获取两个文件夹中的所有文件名
    src_files = set(os.listdir(src_folder))
    dst_files = set(os.listdir(dst_folder))
    
    # 找到只存在于 src_folder 的文件
    unique_files = src_files - dst_files
    
    # 将这些唯一文件复制到目标文件夹
    for file_name in unique_files:
        src_file_path = os.path.join(src_folder, file_name)
        dst_file_path = os.path.join(dst_folder, file_name)
        
        # 仅复制文件，不复制目录（如果需要复制目录，请修改此代码）
        if os.path.isfile(src_file_path):
            shutil.copy2(src_file_path, dst_file_path)
            print(f"Copied {file_name} to {dst_folder}")




def main():
    
    dst_dir = '/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/box_and_points_struct/points_10'
    src_dir = '/home/user512-003/songcw/data/naochuxue/test/label_jpg'
    
    for item in os.listdir(dst_dir):
        print(item)
        folder_A = os.path.join(src_dir,item)
        folder_B = os.path.join(dst_dir,item)

        copy_unique_files(folder_A, folder_B)


if __name__=='__main__':
    main()