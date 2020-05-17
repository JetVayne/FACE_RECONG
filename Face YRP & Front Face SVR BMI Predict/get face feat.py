# coding: utf-8
import os
import cv2
from module.face_feat_helper import FaceFeatHelper
import pandas as pd

img_folder = 'FrontFace'

total = len(os.listdir(img_folder))

meta = pd.read_csv('bmi_map.csv')

df = pd.DataFrame(columns=['id', 'height', 'weight', 'bmi-gt', 'class-gt',
                           'CJWR', 'WHR', 'PAR', 'ES', 'FWR', 'MEH', 'CFR'])

for index, fn in enumerate(os.listdir(img_folder)):
    print('{} / {}'.format(index + 1, total))
    img_path = '{}/{}'.format(img_folder, fn)
    meta_data = list(meta[meta['id'] == fn].values[0])

    image = cv2.imread(img_path)
    ff_helper = FaceFeatHelper(image, fn)
    ff_helper.launch()
    feat = ff_helper.features
    meta_data.extend(feat)
    df.loc[len(df)] = meta_data
    print()

    # break

df.to_csv('xxx.csv', encoding='utf-8', index=False)
