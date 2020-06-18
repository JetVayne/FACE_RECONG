# coding: utf-8
import cv2
import numpy as np
import numpy
from glance.jf_ult.img_proc_tool import ImgProcTool
from glance.face_grid import FaceGrid



class FGridGenerator:
    def __init__(self, image, img_name=None, dest_path=None, strict=False):
        self.scale_range = 10
        self.scale_dist = list()

        self.strict = strict
        self.dest_path = dest_path
        self.image_path = None
        self.image_name = img_name
        self.image = None
        self.result = None

        if isinstance(image, np.ndarray):
            self.image = image
        else:
            self.image_path = image
            self.image = cv2.imread(self.image_path)

        self.roi_top_x = None  # 左上角 X
        self.roi_top_y = None  # 左上角 Y
        self.roi_bot_x = None  # 右下角 X
        self.roi_bot_y = None  # 右下角 X
        self.roi_width = None
        self.roi_height = None
        self.roi_size = None

    def launch(self):
        rotated_img = ImgProcTool.rotater(self.image)
        face_grid = FaceGrid(rotated_img)
        face_grid.fetch()

        self.load_roi_points(face_grid)
        self.scale_dist_helper()

        # output
        if self.strict:
            self.strict_output(rotated_img)
        else:
            self.elastic_output(rotated_img)

    def load_roi_points(self, fg: FaceGrid):
        x_cand = [fg.fg_top_left[0], fg.fg_top_right[0],
                  fg.fg_bot_left[0], fg.fg_bot_right[0]]

        y_cand = [fg.fg_top_left[1], fg.fg_top_right[1],
                  fg.fg_bot_left[1], fg.fg_bot_right[1]]

        # top part
        self.roi_top_x = min(x_cand)
        self.roi_top_y = min(y_cand)

        # bot part
        self.roi_bot_x = max(x_cand)
        self.roi_bot_y = max(y_cand)

        # width, height
        self.roi_width = self.roi_bot_x - self.roi_top_x
        self.roi_height = self.roi_bot_y - self.roi_top_y
        self.roi_size = self.roi_width * self.roi_height

    def scale_dist_helper(self):
        roi_size = self.roi_width * self.roi_height
        base = round(roi_size ** 0.05)

        if base == 0:
            base = 1

        self.scale_dist = list(range(0, base * (self.scale_range + 1), base))

    def get_roi(self, image: numpy.ndarray, scale_dist=10) -> numpy.ndarray:
        temp = image.copy()

        top_x = self.roi_top_x - scale_dist
        top_y = self.roi_top_y - scale_dist
        bot_x = self.roi_bot_x + scale_dist
        bot_y = self.roi_bot_y + scale_dist

        return temp[top_y:bot_y, top_x:bot_x]

    def strict_output(self, image: numpy.ndarray):
        roi = self.get_roi(image, self.scale_dist[self.scale_range])
        cv2.imwrite(self.dest_path, roi)
        self.result = True

    def elastic_output(self, image: numpy.ndarray):
        for scale_dist in reversed(self.scale_dist):
            try:
                roi = self.get_roi(image, scale_dist)
                cv2.imwrite(self.dest_path, roi)
                self.result = True
            except Exception as e:
                print('Crop Failed : {} - scale dist : {}'.format(self.image_path, scale_dist))
                print(e)

            if self.result:
                return
