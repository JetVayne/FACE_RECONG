# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
import numpy as np


class FPYA:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        # calculate
        self.face_pyramid = self.get_face_pyramid()
        self.face_pyramid_area = GeomTool.get_polygon_area(self.face_pyramid)

        self.whole_face = self.get_whole_face()
        self.whole_face_area = GeomTool.get_polygon_area(self.whole_face)

        self.cheek_fat = self.whole_face_area - self.face_pyramid_area

        self.val = self.whole_face_area / self.face_pyramid_area
        self.val_2 = self.face_pyramid_area / self.whole_face_area

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.yellow = (0, 255, 255)
        self.thick = 2

    def get_whole_face(self):
        coords = list()
        for i in range(17):
            lm_id = str(i + 1)
            coords.append(self.landmarks[lm_id])

        return coords

    def get_face_pyramid(self) -> list:
        coords = list()
        lm_ids = [1, 49, 9, 55, 17]
        for lm_id in lm_ids:
            coords.append(self.landmarks[str(lm_id)])

        return coords

    def show(self, dest_path=None):
        temp = self.image.copy()
        wf_pts = np.array(self.whole_face, np.int32)
        wf_pts = wf_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [wf_pts], True, self.red, 3)

        fp_pts = np.array(self.face_pyramid, np.int32)
        fp_pts = fp_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [fp_pts], True, self.green, 2)

        if dest_path:
            cv2.imwrite(dest_path, temp)
            return temp
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)

# from glance.jf_ult.fb_helper import FBHelper
# from glance.jf_ult.lmk_helper import LMKHelper
# path = '1.jpg'
# image = cv2.imread(path)
#
# fb = FBHelper.get_face_block(image)
# lmks = LMKHelper.get_landmarks(image, fb)
#
# fc = FPYA(lmks, image)
# fc.show()
# print(fc.val)
# print(fc.val_2)
