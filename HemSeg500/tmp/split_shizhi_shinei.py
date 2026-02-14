import jsonlines
import json,os
import pandas as pd

file_lable_list = []

def process_json_object(json_str,nii_file_name):
    try:
        # 首先将字符串解析为字典
        json_obj = json.loads(json_str)
        if isinstance(json_obj, dict):
            # 在这里处理字典对象，例如打印
            scan_list = json_obj['scan_list']
            for item in scan_list:
                scan_name = item['scan_name']
                scan_lable = item['scan_lable']
                if scan_name+'.nii.gz' in nii_file_name:
                    label = -1
                    if scan_lable == [0, 0, 1, 0, 0, 1]:
                        label = 1
                    if scan_lable == [0, 1, 0, 0, 0, 1]:
                        label = 0
                    file_lable_list.append([scan_name,label])
                    print(f"This is a dictionary: {scan_name,label}")
            
        else:
            print(f"The parsed object is not a dictionary: {json_obj}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON string: {json_str}, error: {e}")

def read_jsonl_file_line_by_line(filepath,nii_file_name):
    with jsonlines.open(filepath) as reader:
        for line in reader:
            if isinstance(line, str):
                process_json_object(line,nii_file_name)
            else:
                print(f"Line is not a string: {line}")

# 使用示例
# 定义文件路径
file_path = '/home/user512-001/songcw/code/naochuxue/3DCT-SD-IVH-ICH/annotion_2.jsonl'
nii_file_name = os.listdir('/home/user512-001//songcw/data/naochuxue/manual_label')
print(nii_file_name)
read_jsonl_file_line_by_line(file_path,nii_file_name)
# 将数据列表转换为 DataFrame，并指定列名
df = pd.DataFrame(file_lable_list, columns=['filename', 'label'])
# 指定 CSV 文件的保存路径
csv_file_path = 'file_lable.csv'
# 保存为 CSV 文件，包含表头
df.to_csv(csv_file_path, index=False, header=True)
print(f"Data has been saved to {csv_file_path}")


