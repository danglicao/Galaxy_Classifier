import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import json


# 原始 HDF5 文件路径
input_h5_file = 'D:/Ame508_final_data/galaxy_zoo/galaxies_full_label_filtered.h5'
# 输出的训练和测试 HDF5 文件路径
train_h5_file = 'D:/Ame508_final_data/galaxy_zoo/train_data_full_label.h5'
test_h5_file = 'D:/Ame508_final_data/galaxy_zoo/test_data_full_label.h5'

# 读取原始 HDF5 文件
with h5py.File(input_h5_file, 'r') as h5_file:
    # 加载图像数据和标签
    images = h5_file['image_data'][:]
    labels = h5_file['labels'][:]

    # 使用 train_test_split 划分数据集 (80% 训练集，20% 测试集)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size = 0.2, stratify = labels, random_state = 42
    )

    # 保存训练集
    with h5py.File(train_h5_file, 'w') as train_file:
        # 使用相同的变长数据类型存储图像数据
        dt = h5py.special_dtype(vlen = np.uint8)
        train_image_data = train_file.create_dataset("image_data", (len(train_images),), dtype = dt)
        train_label_data = train_file.create_dataset("labels", (len(train_labels),), dtype = 'int')

        for i in range(len(train_images)):
            train_image_data[i] = train_images[i]
            train_label_data[i] = train_labels[i]

        # 保存标签映射
        if 'label_map' in h5_file.attrs:
            train_file.attrs['label_map'] = h5_file.attrs['label_map']

    # 保存测试集
    with h5py.File(test_h5_file, 'w') as test_file:
        # 使用相同的变长数据类型存储图像数据
        test_image_data = test_file.create_dataset("image_data", (len(test_images),), dtype = dt)
        test_label_data = test_file.create_dataset("labels", (len(test_labels),), dtype = 'int')

        for i in range(len(test_images)):
            test_image_data[i] = test_images[i]
            test_label_data[i] = test_labels[i]

        # 保存标签映射
        if 'label_map' in h5_file.attrs:
            test_file.attrs['label_map'] = h5_file.attrs['label_map']

print("Data successfully split into train_data.h5 and test_data.h5.")
