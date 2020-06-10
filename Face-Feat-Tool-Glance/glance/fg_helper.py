# coding: utf-8
import numpy
import os
from glance.jf_ult.fb_helper import FBHelper


class FGHelper:
    def __init__(self, image: numpy.ndarray, save_path='saved_facegrid'):
        self.raw_image = image
        self.save_path = save_path
        if self.save_path == 'saved_facegrid':
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.result = False

        self.face_block = None
        self.fg_width = None
        self.fg_height = None

    def extract_face_grid(self):
        self.face_block = FBHelper.get_face_block(self.raw_image)
        if not self.face_block:
            return

        self.fg_width = self.face_block.width()
        self.fg_height = self.face_block.height()

    def get_crop(self):
        pass

