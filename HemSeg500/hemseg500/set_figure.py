import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('dice_result/attentionunet_dice.csv')

# 创建箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['attentionunet'])

# 添加标题
plt.title('attentionunet')

# 保存图形为PNG文件
plt.savefig('figures/attentionunet.pdf', format='pdf')

# 如果需要保存为其他格式，比如SVG或PDF：
# plt.savefig('boxplot.svg', format='svg')
# plt.savefig('boxplot.pdf', format='pdf')

# 显示图形
plt.show()
