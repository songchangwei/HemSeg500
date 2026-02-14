import nibabel as nib
import numpy as np
from PIL import Image
import pickle,os

def save_slices_and_header(nii_file, output_dir, header_file):
    # Load the NIfTI file
    img = nib.load(nii_file)
    data = img.get_fdata()
    header = img.header
    affine = img.affine

    # Save the header and affine matrix
    with open(header_file, 'wb') as f:
        pickle.dump((header, affine), f)
    print(f"Header and affine matrix saved to: {header_file}")

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each slice as a JPEG
    num_slices = data.shape[2]
    for i in range(num_slices):
        slice_data = data[:, :, i]

        # Normalize the data to 0-255
        normalized_slice = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        slice_image = Image.fromarray(normalized_slice.astype(np.uint8))
        
        # Create a filename with leading zeros
        filename = os.path.join(output_dir, f"{i:04}.jpg")
        slice_image.save(filename)
        print(f"Saved slice {i} as JPEG: {filename}")

def reconstruct_nii(jpeg_dir, header_file, output_nii_file):
    # Load the header and affine matrix
    with open(header_file, 'rb') as f:
        header, affine = pickle.load(f)
    
    # Get sorted list of JPEG files
    jpeg_files = sorted([f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')])

    # Load each slice and add to a list
    slices = []
    for jpeg_file in jpeg_files:
        image_path = os.path.join(jpeg_dir, jpeg_file)
        img = Image.open(image_path)
        slice_data = np.array(img)
        slices.append(slice_data)

    # Stack slices to create a 3D array
    data_3d = np.stack(slices, axis=-1)
    
    # Convert the data back to its original type using header information if necessary
    # Here, we assume the data is originally stored as float32, modify if needed
    data_3d = data_3d.astype(np.float32)

    # Create a NIfTI image
    new_img = nib.Nifti1Image(data_3d, affine, header)

    # Save the new NIfTI file
    nib.save(new_img, output_nii_file)
    print(f"NIfTI file saved to: {output_nii_file}")


def main(nii_dir,output_dir):
    for item in os.listdir(nii_dir):
        nii_file = os.path.join(nii_dir,item)
        jpg_dir = os.path.join(output_dir,item)
        if not os.path.exists(jpg_dir):
            os.mkdir(jpg_dir)
        pkl_file = os.path.join(jpg_dir,'header_affine.pkl')
        print(nii_file, jpg_dir, pkl_file)
        save_slices_and_header(nii_file, jpg_dir, pkl_file)

            
            


if __name__=="__main__":
    main('/home/user512-001/songcw/data/naochuxue/test/image_struct','/home/user512-001/songcw/data/naochuxue/test/image_struct_jpg')
