# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool
import numpy as np


class BCI:
    def __init__(self, landmarks: dict, img=None):
        self.image = img
        self.landmarks = landmarks

        # calculate
        self.bot_chin = self.get_bot_chin()
        self.bot_chin_area = GeomTool.get_polygon_area(self.bot_chin)
        self.bot_chin_perimeter = GeomTool.get_polygon_len(self.bot_chin)

        self.val = self.bot_chin_perimeter / self.bot_chin_area
        self.val_2 = self.bot_chin_area / self.bot_chin_perimeter

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.yellow = (0, 255, 255)
        self.thick = 2

    def get_bot_chin(self) -> list:
        coords = list()
        lm_ids = [5, 7, 8, 9, 10, 11, 13]
        for lm_id in lm_ids:
            coords.append(self.landmarks[str(lm_id)])

        return coords

    def show(self, dest_path=None):
        temp = self.image.copy()
        wf_pts = np.array(self.bot_chin, np.int32)
        wf_pts = wf_pts.reshape((-1, 1, 2))
        cv2.polylines(temp, [wf_pts], True, self.red, 6)
        cv2.polylines(temp, [wf_pts], True, self.green, 2)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
