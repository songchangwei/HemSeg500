import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
import os

def resample_to_target(source_nii, target_nii, output_nii):
    # Load source and target NIfTI files
    source_img = nib.load(source_nii)
    target_img = nib.load(target_nii)

    # Get data, affine, and header from source
    source_data = source_img.get_fdata()
    source_affine = source_img.affine

    # Get affine and shape from target
    target_affine = target_img.affine
    target_shape = target_img.shape

    # Calculate the transformation matrix
    affine_transform_matrix = np.linalg.inv(source_affine).dot(target_affine)

    # Use the affine transformation to resample the source data using nearest neighbor interpolation
    resampled_data = affine_transform(
        source_data,
        affine_transform_matrix[:3, :3],
        offset=affine_transform_matrix[:3, 3],
        output_shape=target_shape,
        order=0  # Nearest neighbor interpolation
    )

    # Ensure the data type is integer (0s and 1s)
    resampled_data = np.round(resampled_data).astype(np.int32)  # Use np.int32 or int

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(resampled_data, target_affine)

    # Save the resampled NIfTI file
    nib.save(new_img, output_nii)
    print(f"Resampled NIfTI file saved to: {output_nii}")


def main(source_nii_dir,target_nii_dir,output_nii_dir):
    for item in os.listdir(target_nii_dir):
        target_nii = os.path.join(target_nii_dir,item)
        source_nii = os.path.join(source_nii_dir,item.split('.')[0]+'_synthseg.nii.gz')
        output_nii = os.path.join(output_nii_dir,item.split('.')[0]+'_synthseg.nii.gz')
        resample_to_target(source_nii,target_nii,output_nii)


if __name__=="__main__":

    # Example usage
    source_nii_dir = '/home/user512-001/songcw/code/naochuxue/SynthSeg/data/test_label'
    target_nii_dir = '/home/user512-001/songcw/data/naochuxue/test/image'
    output_nii_dir = '/home/user512-001/songcw/code/naochuxue/SynthSeg/data/test_label_resample_2'
    main(source_nii_dir, target_nii_dir, output_nii_dir)
