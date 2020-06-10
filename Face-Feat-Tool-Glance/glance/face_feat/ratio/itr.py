# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.lmk_helper import LMKHelper

class ITR:
    def __init__(self, landmarks: dict, img=None):
        self.image = img
        self.landmarks = landmarks
        self.face_scale = LMKHelper.get_face_scale(self.landmarks)

        # calculate
        self.right_jaw = self.landmarks['5']
        self.left_jaw = self.landmarks['13']
        self.jaw_width = GeomTool.get_pt_dist(self.right_jaw, self.left_jaw)

        self.nasal_root = self.landmarks['28']
        self.prj_point = GeomTool.get_prj_point(self.nasal_root, (self.left_jaw, self.right_jaw))
        self.lower_face_height = GeomTool.get_pt_dist(self.nasal_root, self.prj_point)

        self.val = (self.jaw_width * self.face_scale['vrt']) / (self.lower_face_height * self.face_scale['hrz'])
        self.val_2 = (self.jaw_width * self.face_scale['vrt2']) / (self.lower_face_height * self.face_scale['hrz'])
        self.val_3 = (self.jaw_width * self.face_scale['vrt3']) / (self.lower_face_height * self.face_scale['hrz'])
        self.val_4 = (self.jaw_width * self.face_scale['vrt4']) / (self.lower_face_height * self.face_scale['hrz'])

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.nasal_root, self.prj_point, self.red, self.thick)
        cv2.line(temp, self.right_jaw, self.left_jaw, self.green, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
