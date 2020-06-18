# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool


class NLJ:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        # calculate
        self.nose_bot = landmarks['34']
        self.lip_bot = landmarks['58']
        self.jaw_tip = landmarks['9']
        self.nasal_root = landmarks['28']

        self.face_height = GeomTool.get_pt_dist(self.nasal_root, self.jaw_tip)
        self.nose_lip = GeomTool.get_pt_dist(self.nose_bot, self.lip_bot)
        self.lip_jaw = GeomTool.get_pt_dist(self.lip_bot, self.jaw_tip)

        self.val = (self.nose_lip + self.lip_jaw) / self.face_height

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.thick = 2

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.nasal_root, self.jaw_tip, self.red, self.thick)
        cv2.line(temp, self.nose_bot, self.lip_bot, self.green, self.thick)
        cv2.line(temp, self.lip_bot, self.jaw_tip, self.green, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
