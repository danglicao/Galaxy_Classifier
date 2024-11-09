import pandas as pd
import os

# 读取CSV文件
csv_path = 'D:/Ame508_final_data/galaxy_zoo/filtered_galaxy_data.csv'
image_folder = 'D:/Ame508_final_data/galaxy_zoo/images_gz2/images'

df = pd.read_csv(csv_path)


def image_exists(asset_id):
    image_path = os.path.join(image_folder, f"{asset_id}.jpg")
    return os.path.isfile(image_path)


df_filtered = df[df['asset_id'].apply(image_exists)]


output_csv_path = 'D:/Ame508_final_data/galaxy_zoo/labels_has_figs.csv'  # 替换为保存的CSV文件路径
df_filtered.to_csv(output_csv_path, index=False)

print(f"已保存整理后的CSV文件至：{output_csv_path}")
