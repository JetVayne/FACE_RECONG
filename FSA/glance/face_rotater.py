# coding: utf-8
import numpy
import cv2
from glance.face_grid import FaceGrid
from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.geom_tool import GeomTool
from glance.jf_ult.img_proc_tool import ImgProcTool


class FaceRotator:
    def __init__(self, image: numpy.ndarray, image_name=None):
        self.image = image.copy()
        self.image_name = image_name
        self.result = False
        self.face_grid = None

        # output
        self.rt_image = None

    def launch(self):
        # get Face Grid
        status = self.get_face_grid()
        if not status:
            return

        status = self.rotate()
        if not status:
            return

        self.result = True

    def get_face_grid(self) -> bool:
        self.face_grid = FaceGrid(self.image, self.image_name)
        self.face_grid.fetch()

        if self.face_grid.result:
            return True
        else:
            return False

    def rotate(self):
        angle_to_hrzl = GeomTool.get_angle_to_hrzl(self.face_grid.fg_bot_left,
                                                   self.face_grid.fg_bot_right)
        self.rt_image = ImgProcTool.get_rotate_img(self.image, angle_to_hrzl)

        if FBHelper.get_face_block(self.rt_image):
            return True
        else:
            print('Can not get face block from the rotated img - {}'.format(self.image_name))
            return False
