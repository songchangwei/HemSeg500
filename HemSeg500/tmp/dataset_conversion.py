import json,os
import pandas as pd

training_list = []
df = pd.read_csv('dev_set.csv')
print(df.head())
for index, row in df.iterrows():
    print(row['filename']+'_0000.nii.gz')
    training_list.append({"image":"./imagesTr/"+row['filename']+'_0000.nii.gz',"label":"./labelsTr/"+row['filename']+'.nii.gz'})

test_list = []
df = pd.read_csv('test_set.csv')
print(df.head())
for index, row in df.iterrows():
    print(row['filename']+'.nii.gz')
    test_list.append("./imagesTs/"+row['filename']+'_0000.nii.gz')


data = {
  "name": "iphivh",
  "description": "segment iph and ivh dataset",
  "tensorImageSize": "3D",
  "reference": "Paper or website with dataset description",
  "licence": "Dataset license",
  "release": "1.0",
  "channel_names": {
    "0": "CT"
  },
  "labels": {
    "background": "0",
    "ipv_ivh": "1"
  },
  "numTraining": len(training_list),
  "numTest": len(test_list),
  "file_ending": ".nii.gz",
  "training": training_list,
  "test":test_list
}
# 将数据保存为JSON文件
with open('/home/user512-001/songcw/nnunet/nnUNet_raw/Dataset011_CerebralHemorrhage/dataset.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)