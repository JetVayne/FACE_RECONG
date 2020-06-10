# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.lmk_helper import LMKHelper


class TA:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks
        self.face_scale = LMKHelper.get_face_scale(self.landmarks)

        # calculate
        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.cheek_width = GeomTool.get_pt_dist(self.left_cheek, self.right_cheek)

        self.jaw_tip = landmarks['9']
        self.prj_point = GeomTool.get_prj_point(self.jaw_tip, (self.left_cheek, self.right_cheek))
        # Lower Face Height (lft)
        self.lft = GeomTool.get_pt_dist(self.jaw_tip, self.prj_point)

        self.val = (self.cheek_width * self.lft) / (self.face_scale['hrz'] * self.face_scale['vrt'])
        self.val_2 = (self.cheek_width * self.lft) / (self.face_scale['hrz'] * self.face_scale['vrt2'])
        self.val_3 = (self.cheek_width * self.lft) / (self.face_scale['hrz'] * self.face_scale['vrt3'])
        self.val_4 = (self.cheek_width * self.lft) / (self.face_scale['hrz'] * self.face_scale['vrt4'])

        # display
        self.red = (0, 0, 255)
        self.tiff_blue = (208, 216, 129)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.left_cheek, self.right_cheek, self.tiff_blue, self.thick)
        cv2.line(temp, self.jaw_tip, self.prj_point, self.tiff_blue, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
