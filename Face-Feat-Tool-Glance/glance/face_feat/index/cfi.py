# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.lmk_helper import LMKHelper
import numpy as np


class CFI:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks
        self.face_scale = LMKHelper.get_face_scale(self.landmarks)

        # calculate
        self.face_pyramid = self.get_face_pyramid()
        self.face_pyramid_area = GeomTool.get_polygon_area(self.face_pyramid)

        self.whole_face = self.get_whole_face()
        self.whole_face_area = GeomTool.get_polygon_area(self.whole_face)

        # basic point
        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.jaw_tip = self.landmarks['9']

        # support point
        self.prj_point = GeomTool.get_prj_point(self.jaw_tip, (self.left_cheek, self.right_cheek))

        # the distance between horizontal line and jaw tip
        self.prj_jaw = GeomTool.get_pt_dist(self.jaw_tip, self.prj_point)

        # the distance between nasal root and jaw tip
        self.nasal_root = self.landmarks['28']
        self.nasal_jaw = GeomTool.get_pt_dist(self.nasal_root, self.jaw_tip)

        self.cheek_fat = self.whole_face_area - self.face_pyramid_area
        self.cheek_fat_ratio = self.cheek_fat / self.whole_face_area

        # CFR val
        self.val = self.cheek_fat_ratio / self.prj_jaw
        self.val_2 = self.cheek_fat_ratio / self.nasal_jaw

        self.val_3 = (self.cheek_fat_ratio * self.face_scale['vrt']) / self.prj_jaw
        self.val_4 = (self.cheek_fat_ratio * self.face_scale['vrt2']) / self.prj_jaw

        self.val_5 = (self.cheek_fat_ratio * self.face_scale['vrt']) / self.nasal_jaw
        self.val_6 = (self.cheek_fat_ratio * self.face_scale['vrt2']) / self.nasal_jaw

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.purple = (128, 0, 128)
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


        cv2.line(temp, self.prj_point, self.jaw_tip, self.purple, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
