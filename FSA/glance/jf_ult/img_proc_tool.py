# coding: utf-8
import cv2
import numpy
from glance.face_grid import FaceGrid
from glance.jf_ult.geom_tool import GeomTool


class ImgProcTool:
    @staticmethod
    def img_read(img_path: str):
        return cv2.imdecode(numpy.fromfile(img_path, dtype=numpy.uint8), -1)

    @staticmethod
    def get_rotate_img(image: numpy.ndarray, angle: float, center=None, scale=1.0) -> numpy.ndarray:
        temp = image.copy()
        # get img size
        h, w = temp.shape[:2]

        # 寬高除 2 之座標，當作旋轉中心
        if center is None:
            center = (w / 2, h / 2)

        # rotate
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(temp, M, (w, h))

        return rotated

    @staticmethod  # not good
    def rotater(image: numpy.ndarray, mode='fg', angle=None):
        temp = image.copy()
        if mode == 'fg':
            fg = FaceGrid(temp)
            fg.fetch()
            angle_to_hrzl = GeomTool.get_angle_to_hrzl(fg.fg_bot_left, fg.fg_bot_right)
            return ImgProcTool.get_rotate_img(temp, angle_to_hrzl)
