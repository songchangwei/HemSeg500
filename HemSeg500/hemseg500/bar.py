import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
# 从 Excel 文件读取数据
data = pd.read_excel('output3.xlsx', usecols=['一致性', '可靠性', '专业性', '总分'])

# 将读取的数据转换成列表形式，适用于箱线图的数据输入
all_data = [data[column].dropna() for column in data]

# 创建一个图和一个轴
fig, ax = plt.subplots()

# 绘制箱线图
bplot = ax.boxplot(all_data,
                   vert=True,  # 垂直箱线图
                   patch_artist=True,  # 填充颜色
                   showmeans=True,  # 显示均值线
                   meanline=True)  # 用线表示均值

# 颜色填充
colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# 添加水平网格线
ax.yaxis.grid(True)

# 设置x轴和y轴的标签
ax.set_xticks([y + 1 for y in range(len(all_data))])
ax.set_xticklabels(['一致性', '可靠性', '专业性', '总体分'], fontsize=14)

# 设置刻度字体大小
ax.tick_params(axis='both', labelsize=14)

# 动态调整文本标注的位置以避免重叠并确保较大值在上
offset_adjustment = 0.03  # 基本偏移量
for i, median_line in enumerate(bplot['medians']):
    median_x, median_y = median_line.get_xydata()[1]  # 中位数位置
    mean_x, mean_y = bplot['means'][i].get_xydata()[1]  # 均值位置

    if median_y > mean_y:
        median_offset = offset_adjustment
        mean_offset = -offset_adjustment
    else:
        median_offset = -offset_adjustment
        mean_offset = offset_adjustment

    # 添加均值标注
    ax.annotate(f'mean={mean_y:.2f}', xy=(mean_x, mean_y), xytext=(mean_x, mean_y + mean_offset),
                textcoords='data', ha='center', va='bottom' if mean_offset > 0 else 'top', fontsize=12)

    # 添加中位数标注
    ax.annotate(f'median={median_y:.2f}', xy=(median_x, median_y), xytext=(median_x, median_y + median_offset),
                textcoords='data', ha='center', va='bottom' if median_offset > 0 else 'top', fontsize=12)

# 显示图形
plt.tight_layout()
plt.savefig('human-eval-GPT.pdf', dpi=600)  # 保存为 PDF 文件
plt.show()
