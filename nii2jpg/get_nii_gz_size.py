import nibabel as nib
import os

def get_size(nii_file):

    # 读取 NIfTI 文件
    img = nib.load(nii_file)

    # 获取数据的形状
    shape = img.shape
    print('Data shape:', shape)

    # 获取体素尺寸（像素间距）
    voxel_size = img.header.get_zooms()
    print('Voxel size:', voxel_size)

    # 计算数据在内存中的大小（以字节为单位）
    data_size = img.get_fdata().nbytes
    print('Data size in memory (bytes):', data_size)


def main(source_nii_dir,target_nii_dir,output_nii_dir):
    for item in os.listdir(target_nii_dir):
        target_nii = os.path.join(target_nii_dir,item)
        source_nii = os.path.join(source_nii_dir,item.split('.')[0]+'_synthseg.nii.gz')
        output_nii = os.path.join(output_nii_dir,item.split('.')[0]+'_synthseg.nii.gz')
        for item in [source_nii,target_nii,output_nii]:
            get_size(item)


if __name__=="__main__":

    # Example usage
    source_nii_dir = '/home/user512-003/songcw/code/naochuxue/SynthSeg/data/test_label'
    target_nii_dir = '/home/user512-003/songcw/data/naochuxue/test/image'
    output_nii_dir = '/home/user512-003/songcw/code/naochuxue/SynthSeg/data/test_label_resample'
    main(source_nii_dir, target_nii_dir, output_nii_dir)