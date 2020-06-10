# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.lmk_helper import LMKHelper


class FMWR:
    def __init__(self, landmarks: dict, img=None):
        self.image = img
        self.landmarks = landmarks
        self.face_scale = LMKHelper.get_face_scale(self.landmarks)

        # calculate
        self.right_jaw = self.landmarks['4']
        self.left_jaw = self.landmarks['14']
        self.jaw_width = GeomTool.get_pt_dist(self.right_jaw, self.left_jaw)

        self.mouth_left = self.landmarks['49']
        self.mouth_right = self.landmarks['55']
        self.mouth_width = GeomTool.get_pt_dist(self.mouth_left, self.mouth_right)

        self.val = self.jaw_width / self.mouth_width
        self.val_2 = (self.jaw_width * self.mouth_width) / (self.face_scale['hrz'] ** 2)

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.right_jaw, self.left_jaw, self.green, self.thick)
        cv2.line(temp, self.mouth_left, self.mouth_right, self.red, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)

