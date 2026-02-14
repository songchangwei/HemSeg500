import pandas as pd
import os

# 获取目录中所有CSV文件的文件名
csv_files = ['attentionunet_dice.csv','segresnet_dice.csv','swinunetr_dice.csv', 'unet3d_dice.csv','unetplusplus_dice.csv','unetr_dice.csv','vnet_dice.csv','nnunet_dice.csv','sam2_box_dice.csv', 'sam_points_dice.csv', 'sam2_box_and_points_dice.csv']

# 读取并合并所有CSV文件
dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join('dice_result',file))  # 读取每个CSV文件
    dfs.append(df)  # 将每个DataFrame加入列表

# 按列合并所有DataFrame
merged_df = pd.concat(dfs, axis=1)

# 将合并后的DataFrame保存为一个新的CSV文件
merged_df.to_csv('dice_result/merged_data_by_columns.csv', index=False)

print("按列合并完成，文件已保存为 'dice_result/merged_data_by_columns.csv'")
