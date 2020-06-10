# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
import numpy as np


class CFA:
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

        # CFR val
        self.val = self.cheek_fat / self.whole_face_area
        self.val_2 = self.whole_face_area / self.cheek_fat

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.yellow = (0, 255, 255)
        self.thick = 2

    def get_whole_face(self):
        coords = []
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

        left_cheek_coords = list()
        for i in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '49']:
            left_cheek_coords.append(self.landmarks[i])
        lc_pts = np.array(left_cheek_coords, np.int32)
        lc_pts = lc_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [lc_pts], True, self.green, 1)

        right_cheek_coords = list()
        for i in ['9', '10', '11', '12', '13', '14', '15', '16', '17', '55']:
            right_cheek_coords.append(self.landmarks[i])

        rc_pts = np.array(right_cheek_coords, np.int32)
        rc_pts = rc_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [rc_pts], True, self.green, 1)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
