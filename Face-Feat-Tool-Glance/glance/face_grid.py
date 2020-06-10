# coding: utf-8
import cv2
import numpy

from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper


class FaceGrid:
    def __init__(self, image, img_name=None):
        self.scale_dist = 10
        self.image_path = None
        self.image_name = img_name
        self.image = None
        self.result = False

        if isinstance(image, numpy.ndarray):
            self.image = image
        else:
            self.image_path = image
            self.image = cv2.imread(self.image_path)

        self.face_block = None
        self.landmarks = dict()

        # support point
        self.reb_top = None
        self.leb_top = None
        self.cheek_right = None
        self.cheek_left = None

        # face grid points
        self.fg_top_left = None
        self.fg_top_right = None
        self.fg_bot_left = None
        self.fg_bot_right = None

        # output
        self.top_x = None
        self.top_y = None
        self.bot_x = None
        self.bot_y = None

        self.width = None
        self.height = None
        self.size = None

    def fetch(self):
        self.face_block = FBHelper.get_face_block(self.image)
        if self.face_block is None:
            print('[WARN] - Failed to fetch face block.')
            self.result = False
            return

        self.landmarks = LMKHelper.get_landmarks(self.image, self.face_block)
        self.load_sup_points()
        self.load_fg_points()
        self.load_roi_points()

        self.result = True

    def load_sup_points(self):
        self.reb_top = LMKHelper.get_reb_top(self.landmarks)
        self.leb_top = LMKHelper.get_leb_top(self.landmarks)
        self.cheek_right, self.cheek_left = LMKHelper.get_edge_points(self.landmarks,
                                                                      (self.reb_top, self.leb_top))

    def load_fg_points(self):
        # face grid top right
        self.fg_top_right = LMKHelper.get_fg_top_right(self.cheek_left,
                                                       (self.reb_top, self.leb_top))
        # face grid top left
        self.fg_top_left = LMKHelper.get_fg_top_left(self.cheek_right,
                                                     (self.reb_top, self.leb_top))
        # face grid down right 
        self.fg_bot_right = LMKHelper.get_fg_bot_right(self.landmarks['9'],
                                                       (self.fg_top_right, self.cheek_left))
        # face grid down left
        self.fg_bot_left = LMKHelper.get_fg_bot_left(self.landmarks['9'],
                                                     (self.fg_top_left, self.cheek_right))

    def load_roi_points(self):
        x_cand = [self.fg_top_left[0], self.fg_top_right[0],
                  self.fg_bot_left[0], self.fg_bot_right[0]]

        y_cand = [self.fg_top_left[1], self.fg_top_right[1],
                  self.fg_bot_left[1], self.fg_bot_right[1]]

        # top part
        self.top_x = min(x_cand)
        self.top_y = min(y_cand)

        # bot part
        self.bot_x = max(x_cand)
        self.bot_y = max(y_cand)

        # width, height
        self.width = self.bot_x - self.top_x
        self.height = self.bot_y - self.top_y
        self.size = self.width * self.height
