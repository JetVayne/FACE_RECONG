# coding: utf-8
import cv2
from module.jf_ult.geom_tool import GeomTool
import numpy as np


class PAR:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        ## calculate
        self.face_coords = self.get_face_coords()
        self.face_area = GeomTool.get_polygon_area(self.face_coords)
        self.face_perimeter = GeomTool.get_polygon_len(self.face_coords)

        # print('周長: {} / 面積: {}'.format(self.face_perimeter, self.face_area))

        self.val = self.face_perimeter / self.face_area

        # display
        self.red = (0, 0, 255)
        self.thick = 2

    def get_face_coords(self):
        coords = []
        for i in range(17):
            lm_id = str(i + 1)
            coords.append(self.landmarks[lm_id])

        return coords

    def show(self, dest_path=None):
        temp = self.image.copy()
        wf_pts = np.array(self.face_coords, np.int32)
        wf_pts = wf_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [wf_pts], True, self.red, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
