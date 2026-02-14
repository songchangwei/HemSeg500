import nibabel as nib
import numpy as np
import os

def brain_by_mask(source_file,mask_file,target_file):
    # 加载 NIfTI 文件
    a_img = nib.load(source_file)
    mask_img = nib.load(mask_file)

    # 获取数据数组
    a_data = a_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # 确保掩码是二值的
    mask_data = mask_data > 0

    # 创建一个与原始图像相同形状的空数组
    extracted_data_full = np.zeros_like(a_data)

    # 将提取的数据放入这个空数组的掩码位置
    extracted_data_full[mask_data] = a_data[mask_data]

    # 创建新的 NIfTI 图像
    extracted_img = nib.Nifti1Image(extracted_data_full, affine=a_img.affine, header=a_img.header)

    # 保存提取的数据为新的 NIfTI 文件
    nib.save(extracted_img, target_file)

    print("提取的数据已保存为"+target_file)


def main():
    source_folder = '/home/user512-001/songcw/data/naochuxue/test/image'
    mask_folder = '/home/user512-001/songcw/code/naochuxue/SynthSeg/data/test_label_mask'
    target_folder = '/home/user512-001/songcw/data/naochuxue/test/image_struct'
    for item in os.listdir(source_folder):
        source_file = os.path.join(source_folder,item)
        mask_file = os.path.join(mask_folder,item.split('.')[0]+'_synthseg.nii.gz')
        target_file = os.path.join(target_folder,item)
        print(source_file,mask_file,target_file)
        brain_by_mask(source_file,mask_file,target_file)

if __name__=='__main__':
    main()