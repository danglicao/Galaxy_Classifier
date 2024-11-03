import pandas as pd
import re

df = pd.read_csv('D:/Ame508_final_data/galaxy_zoo/zoo2_whole_data.csv')


# df['galaxy_type'] = df['gz2class'].apply(lambda x: re.match(r'(S|SB|E)', x).group() if re.match(r'(S|SB|E)', x) else None)
df['galaxy_type'] = df['gz2class'].apply(lambda x: re.match(r'^(SB|S|E|I)', x).group() if re.match(r'^(SB|S|E)', x) else None)
new_df = df[['asset_id', 'galaxy_type']].dropna()

new_df.to_csv('D:/Ame508_final_data/galaxy_zoo/filtered_galaxy_data.csv', index=False)
# df.to_csv('D:/Ame508_final_data/galaxy_zoo/zoo2_whole_data.csv', index=False)

print("filtered_galaxy_data.csv")
