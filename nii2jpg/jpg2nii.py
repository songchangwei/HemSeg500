import os
import numpy as np
from PIL import Image
import pickle
import nibabel as nib

# 读取JPEG切片文件
def load_slices(slice_dir):
    slices = []
    for filename in sorted(os.listdir(slice_dir)):
        if filename.endswith(".jpg"):
            slice_path = os.path.join(slice_dir, filename)
            image = Image.open(slice_path)
            slices.append(np.array(image))
    return np.stack(slices, axis=-1)

# 读取头信息和仿射矩阵
def load_header_affine(pkl_path):
    with open(pkl_path, 'rb') as f:
        header, affine = pickle.load(f)
    return header, affine

# 组装NIfTI文件
def reconstruct_nifti(slices, header, affine):
    img = nib.Nifti1Image(slices, affine, header=header)
    return img




def main():
    # 路径设置
    src_dir = '/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/box_and_points_struct/points_10'  
    dst_dir = '/home/user512-003/songcw/code/naochuxue/sam2_for_ICH/output_sam2_masks/box_and_points/points_10'  

    for item in os.listdir(src_dir):
        slice_dir = os.path.join(src_dir,item)
        pkl_path = os.path.join(src_dir,item,'header_affine.pkl')
        output_nifti = os.path.join(dst_dir,item)
        slices = load_slices(slice_dir)
        header, affine = load_header_affine(pkl_path)
        nifti_img = reconstruct_nifti(slices, header, affine)
        nib.save(nifti_img, output_nifti)

        print(f"NIfTI file saved to {output_nifti}")
    
if __name__=='__main__':
    main()
