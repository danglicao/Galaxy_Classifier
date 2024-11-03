import pandas as pd
import re

path = 'D:/Ame508_final_data/galaxy_zoo/zoo2_whole_data.csv'

df = pd.read_csv(path)


def parse_galaxy_class(class_name):
    # 初始化字典来存储解析结果
    parsed = {
        'main_type': None,
        'sub_type': None,
        'modifiers': []
    }

    # 提取括号内的内容作为修饰符
    modifiers = re.findall(r'\((.*?)\)', class_name)
    if modifiers:
        parsed['modifiers'].extend(modifiers)

    # 移除括号内的内容
    class_name = re.sub(r'\(.*?\)', '', class_name)

    # 提取主要类型（如 E、S、SB、I）
    main_type_match = re.match(r'^(E|SB|S|I)', class_name)
    if main_type_match:
        parsed['main_type'] = main_type_match.group()
        class_name = class_name[len(parsed['main_type']):]
    else:
        parsed['main_type'] = 'Unknown'

    # 提取亚型（如 a、b、c、d、数字等）
    sub_type_match = re.match(r'([a-dm]+|\d+)', class_name, re.IGNORECASE)
    if sub_type_match:
        parsed['sub_type'] = sub_type_match.group()
        class_name = class_name[len(parsed['sub_type']):]

    # 提取特殊符号作为修饰符
    symbols = re.findall(r'[\+\-\?]', class_name)
    if symbols:
        parsed['modifiers'].extend(symbols)

    return parsed


# 应用解析函数
parsed_data = df['gz2class'].apply(parse_galaxy_class)
parsed_df = pd.json_normalize(parsed_data)

# 将解析结果添加到原始DataFrame
df = pd.concat([df, parsed_df], axis=1)

columns_to_save = ['asset_id', 'main_type', 'sub_type','modifiers']
final_df = df[columns_to_save]
# 查看结果
print(df.head())
new_df = df[columns_to_save].dropna()
# 保存解析后的数据
# df.to_csv('parsed_galaxy_data.csv', index = False)
new_df.to_csv('D:/Ame508_final_data/galaxy_zoo/distract_label.csv', index=False)
print("已生成解析后的CSV文件：parsed_galaxy_data.csv")