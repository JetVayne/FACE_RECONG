# coding: utf-8
import cv2
from module.jf_ult.geom_tool import GeomTool
import numpy as np


class CFR:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        ## calculate
        self.face_pyramid = self.get_face_pyramid()
        self.face_pyramid_area = GeomTool.get_polygon_area(self.face_pyramid)

        self.whole_face = self.get_whole_face()
        self.whole_face_area = GeomTool.get_polygon_area(self.whole_face)

        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.jaw_tip = self.landmarks['9']
        self.prj_point = GeomTool.get_prj_point(self.jaw_tip, (self.left_cheek, self.right_cheek))

        # face vertical line
        self.face_vl = GeomTool.get_pt_dist(self.prj_point, self.jaw_tip)

        # nasal root to jaw
        self.nasal_root = self.landmarks['28']
        self.nasal_to_jaw = GeomTool.get_pt_dist(self.nasal_root, self.jaw_tip)

        # face height ratio
        self.fhr = self.face_vl / self.nasal_to_jaw
        # face height
        self.face_height = GeomTool.get_pt_dist(self.jaw_tip, self.prj_point)

        # print(self.face_pyramid)
        # print('整臉: {}'.format(self.whole_face_area))
        # print('三角: {}'.format(self.face_pyramid_area))
        # print('長: {}'.format(self.fhr))

        # self.val = (self.whole_face_area / self.face_pyramid_area) / self.face_height
        self.val = 100 * (self.whole_face_area - self.face_pyramid_area) / self.whole_face_area / self.fhr
        # self.val = 1000 * (self.whole_face_area - self.face_pyramid_area) / self.whole_face_area

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
        cv2.polylines(temp, [wf_pts], True, self.red, self.thick)

        fp_pts = np.array(self.face_pyramid, np.int32)
        fp_pts = fp_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [fp_pts], True, self.yellow, 1)

        cv2.line(temp, self.prj_point, self.jaw_tip, self.green, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
