# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.lmk_helper import LMKHelper


class UCA:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks
        self.face_scale = LMKHelper.get_face_scale(self.landmarks)

        # calculate
        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.cheek_width = GeomTool.get_pt_dist(self.left_cheek, self.right_cheek)

        self.nasal_root = self.landmarks['28']
        self.lip_top = self.landmarks['52']
        # Lower Face Height (lft)
        self.nlh = GeomTool.get_pt_dist(self.nasal_root, self.lip_top)

        self.val = (self.cheek_width * self.nlh) / (self.face_scale['hrz'] * self.face_scale['vrt'])
        self.val_2 = (self.cheek_width * self.nlh) / (self.face_scale['hrz'] * self.face_scale['vrt2'])
        self.val_3 = (self.cheek_width * self.nlh) / (self.face_scale['hrz'] * self.face_scale['vrt3'])
        self.val_4 = (self.cheek_width * self.nlh) / (self.face_scale['hrz'] * self.face_scale['vrt4'])

        # display
        self.red = (0, 0, 255)
        self.tiff_blue = (208, 216, 129)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.left_cheek, self.right_cheek, self.tiff_blue, self.thick)
        cv2.line(temp, self.nasal_root, self.lip_top, self.tiff_blue, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
            return temp
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)

