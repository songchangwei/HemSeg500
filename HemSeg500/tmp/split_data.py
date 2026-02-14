import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv('file_lable.csv')

# 检查数据格式
print(df.head())

# 分别对每个标签值 (0 和 1) 进行 4:1 的随机划分
train_list = []
test_list = []

for label in df['label'].unique():
    subset = df[df['label'] == label]
    train_subset, test_subset = train_test_split(subset, test_size=0.2, random_state=42)
    train_list.append(train_subset)
    test_list.append(test_subset)

# 合并数据集
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# 保存结果到新的 CSV 文件
train_df.to_csv('dev_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

print("Data has been split and saved to 'dev_set.csv' and 'test_set.csv'")
