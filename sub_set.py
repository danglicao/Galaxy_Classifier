import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
csv_path = 'D:/Ame508_final_data/galaxy_zoo/labels_has_figs.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_path)

# 设定子集的大小，例如 2000 条
subset_size = 5000

# 使用 train_test_split 按 label 进行分层采样
df_subset, _ = train_test_split(df, train_size=subset_size, stratify=df['galaxy_type'], random_state=42)

# 保存子集为新的 CSV 文件
subset_csv_path = 'D:/Ame508_final_data/galaxy_zoo/subset_labels.csv'  # 替换为保存子集的 CSV 文件路径
df_subset.to_csv(subset_csv_path, index=False)

print(f"已生成包含 {subset_size} 条数据的子集，保存至：{subset_csv_path}")
