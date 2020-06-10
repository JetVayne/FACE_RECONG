# coding: utf-8
import cv2
from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper
from glance.jf_ult.display_tool import DisplayTool
import os


class OutputLMKS:
    def __init__(self, img_input, mode=None, dest_path=None):
        self.img_path = None
        self.image = None
        if isinstance(img_input, str):
            self.img_path = img_input
            self.image = cv2.imread(img_input)
        else:
            self.image = img_input

        self.mode = mode
        self.dest_path = dest_path


        self.face_block = None
        self.landmarks = dict()

    def launch(self):
        self.load_face_block()
        self.load_landmarks()

        if self.mode == 'g':
            return DisplayTool.show_landmarks(self.image, self.landmarks, rt_img=True)

        if self.mode:
            DisplayTool.show_landmarks(self.image, self.landmarks, self.dest_path)
        else:
            DisplayTool.show_landmarks(self.image, self.landmarks)

    # def load_image(self):
    #     if os.path.exists(self.img_path):
    #         self.image = cv2.imread(self.img_path)
    #     else:
    #         print('!!!!!!!!!!!!!!!!!!!!!!!!!')

    def load_face_block(self):
        self.face_block = FBHelper.get_face_block(self.image)

    def load_landmarks(self):
        self.landmarks = LMKHelper.get_landmarks(self.image, self.face_block)
