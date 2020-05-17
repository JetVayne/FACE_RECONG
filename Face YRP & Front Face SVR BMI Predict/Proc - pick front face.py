# coding: utf-8
import os

import cv2

from module.face_yrp import FaceYRP

folder_path = 'IMG/DATA/A'
# folder_path = 'IMG/MakeDoc/ttg-raw'


total = len(os.listdir(folder_path))

threshold = 10


def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated


for index, fn in enumerate(os.listdir(folder_path)):
    print('{} / {}'.format(index + 1, total))
    img_path = '{}/{}'.format(folder_path, fn)
    if os.path.exists(img_path):
        raw_img = cv2.imread(img_path)
        try:
            face_yrp = FaceYRP(raw_img, image_name=fn)
            face_yrp.launch()
            if face_yrp.result:
                print(face_yrp.roll, face_yrp.pitch, face_yrp.yaw)
                if face_yrp.roll and -threshold < face_yrp.pitch < threshold and -threshold < face_yrp.yaw < threshold:
                    dest_output_path = 'IMG/MakeDoc/output'
                    dest_path = '{}/{}'.format(dest_output_path, fn)
                    temp = raw_img.copy()
                    temp = rotate(temp, - face_yrp.roll)
                    cv2.imwrite(dest_path, temp)

            # # output yrp line image
            # dest_path = '{}/{}'.format(dest_output_path, fn)
            # face_yrp = FaceYRP(raw_img, image_name=fn, mode='o', dest_path=dest_path)
            # face_yrp.launch()

        except Exception as e:
            print(e)
            print('[WARN] - error Image - {}'.format(img_path))

    else:
        print('!!!!', img_path)
