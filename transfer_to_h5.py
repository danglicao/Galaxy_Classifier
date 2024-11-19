import h5py
import pandas as pd
import os
from PIL import Image
import numpy as np
import json
import cv2

csv_path = 'D:/Ame508_final_data/galaxy_zoo/zoo2_whole_data.csv'
image_folder = 'D:/Ame508_final_data/galaxy_zoo/images_gz2/images'
output_h5_file = 'D:/Ame508_final_data/galaxy_zoo/galaxies_full_label.h5'

data = pd.read_csv(csv_path)

labels = data['gz2class'].unique()
label_map = {label: idx for idx, label in enumerate(labels)}

num_images = len(data)
print(f"Number of images to store: {num_images}")

img_shape = (424, 424, 3)

dt = h5py.special_dtype(vlen=np.uint8)
with h5py.File(output_h5_file, 'w') as h5_file:
    image_data = h5_file.create_dataset("image_data", (num_images,), dtype=dt)
    label_data = h5_file.create_dataset("labels", (num_images,), dtype='int')

    for i in range(num_images):
        asset_id = data['asset_id'].iloc[i]
        galaxy_type = data['gz2class'].iloc[i]
        image_path = os.path.join(image_folder, f"{asset_id}.jpg")
        try:
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            image_data[i] = np.frombuffer(img_bytes, dtype='uint8')
            label_data[i] = label_map[galaxy_type]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

# with h5py.File(output_h5_file, 'w') as h5_file:
#     image_data = h5_file.create_dataset(
#         "image_data",
#         (num_images, *img_shape),
#         dtype='uint8',
#         compression='gzip',
#         compression_opts=4,
#         chunks=(100, *img_shape)  # 根据需要调整分块大小
#     )
#     label_data = h5_file.create_dataset(
#         "labels",
#         (num_images,),
#         dtype='int',
#         compression='gzip',
#         compression_opts=4
#     )
#     batch_size = 1000
#     for i in range(0, num_images, batch_size):
#         batch_images = []
#         batch_labels = []
#         for j in range(i, min(i + batch_size, num_images)):
#             asset_id = data['asset_id'].iloc[j]
#             galaxy_type = data['galaxy_type'].iloc[j]
#             image_path = os.path.join(image_folder, f"{asset_id}.jpg")
#             try:
#                 img = cv2.imread(image_path)
#                 img = cv2.resize(img, (424, 424))
#                 img_array = img.astype('uint8')
#
#                 if img_array.shape != img_shape:
#                     img_array = img_array[:424, :424, :3]
#                 batch_images.append(img_array)
#                 batch_labels.append(label_map[galaxy_type])
#             except Exception as e:
#                 print(f"Error loading image {image_path}: {e}")
#                 continue
#         image_data[i:i + batch_size] = np.array(batch_images)
#         label_data[i:i + batch_size] = np.array(batch_labels)

    # for i, (asset_id, galaxy_type) in enumerate(zip(data['asset_id'], data['galaxy_type'])):
    #     image_path = os.path.join(image_folder, f"{asset_id}.jpg")
    #     try:
    #         # with Image.open(image_path) as img:
    #             # img = img.resize((424, 424))
    #             # img_array = np.array(img, dtype='uint8')  # 确保数据类型正确
    #         img = cv2.imread(image_path)
    #         img = cv2.resize(img, (424, 424))
    #         img_array = img.astype('uint8')
    #
    #         if img_array.shape != img_shape:
    #             img_array = img_array[:424, :424, :3]  # 确保形状一致
    #
    #         image_data[i] = img_array
    #         label_data[i] = label_map[galaxy_type]
    #     except Exception as e:
    #         print(f"Error loading image {image_path}: {e}")
    #         continue
    #
    # h5_file.attrs['label_map'] = json.dumps(label_map)

print("Data successfully saved to HDF5 file.")
