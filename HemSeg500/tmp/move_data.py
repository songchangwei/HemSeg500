import pandas as pd
import shutil

df = pd.read_csv('tmp/test_set.csv')
print(f'Header: {df.columns.tolist()}')

# 按行迭代数据
for index, row in df.iterrows():
    print(row['filename']+'.nii.gz')
    source_image_file = '/home/user512-001/songcw/data/naochuxue/image/'+row['filename']+'.nii.gz'
    target_iamge_file = '/home/user512-001/songcw/data/naochuxue/test/image/'+row['filename']+'.nii.gz'
    source_label_file = '/home/user512-001/songcw/data/naochuxue/manual_label/'+row['filename']+'.nii.gz'
    target_label_file = '/home/user512-001/songcw/data/naochuxue/test/label/'+row['filename']+'.nii.gz'
    shutil.copy2(source_image_file,target_iamge_file)
    shutil.copy2(source_label_file,target_label_file)
