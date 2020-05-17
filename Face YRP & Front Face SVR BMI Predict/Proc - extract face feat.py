# coding: utf-8
import os
import cv2
from module.face_feat_helper import FaceFeatHelper
import pandas as pd

folder_path = 'IMG/DATA/A'

total = len(os.listdir(folder_path))

meta = pd.read_csv('bmi_map.csv')

df = pd.DataFrame(columns=['id', 'height', 'weight', 'bmi-gt', 'class-gt',
                           'CJWR', 'WHR', 'PAR', 'ES', 'FWR', 'MEH', 'CFR'])


for index, fn in enumerate(os.listdir(folder_path)):
    print('{} / {}'.format(index + 1, total))
    img_path = '{}/{}'.format(folder_path, fn)
    meta_data = list(meta[meta['id'] == fn].values[0])

    try:
        image = cv2.imread(img_path)
        ff_helper = FaceFeatHelper(image, fn)
        ff_helper.launch()
        feat = ff_helper.features
        meta_data.extend(feat)
        df.loc[len(df)] = meta_data
    except Exception as e:
        print(e)
        print('[warn] - 處理失敗 - {}'.format(img_path))

    # break

# df.to_csv('front_face.csv', encoding='utf-8', index=False)

# # output face feat schematic diagram
# for index, fn in enumerate(os.listdir(folder_path)):
#     print('{} / {}'.format(index + 1, total))
#     img_path = '{}/{}'.format(folder_path, fn)
#     if os.path.exists(img_path):
#         raw_img = cv2.imread(img_path)
#         dest_output_path = 'IMG/MakeDoc/output'
#         ff_helper = FaceFeatHelper(raw_img, fn, mode='o', dest_folder_path=dest_output_path)
#         ff_helper.launch()
