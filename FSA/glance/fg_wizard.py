# coding: utf-8
import cv2
import numpy as np
import numpy
from glance.jf_ult.img_proc_tool import ImgProcTool
from glance.jf_ult.log_tool import LogTool
from glance.face_grid import FaceGrid
from glance.jf_ult.fb_helper import FBHelper


class FGWizard:
    def __init__(self, image: numpy.ndarray, img_name=None, dest_path=None, strict=False):
        self.scale_range = 10
        self.scale_dist = list()

        self.strict = strict
        self.dest_path = dest_path

        self.image = image.copy()
        self.image_name = img_name

        self.result = False

        # roi_width, roi_height, roi_size, roi_image_w, roi_image_h, roi_image_size
        self.record = list()

        self.face_grid = None

        self.roi_top_x = None  # 左上角 X
        self.roi_top_y = None  # 左上角 Y
        self.roi_bot_x = None  # 右下角 X
        self.roi_bot_y = None  # 右下角 X

        # roi meta
        self.roi_width = None
        self.roi_height = None
        self.roi_size = None

        self.roi_image = None

        # roi image meta
        self.roi_image_w = None
        self.roi_image_h = None
        self.roi_image_size = None

    def launch(self):
        # get face grid
        status = self.get_face_grid()
        if not status:
            return

        # load roi points
        status = self.load_roi_points(self.face_grid)
        if not status:
            print('load roi 出問題!')
            return

        # 設定縮放距離
        status = self.scale_dist_helper()
        if not status:
            print('設定縮放距離出問題!')
            return

        # output work
        status = self.output_handler()
        if not status:
            msg = '{}'.format(self.image_name)
            LogTool.update_slog('fg-log-error.txt', msg, 'a')
            return

        # collect record
        status = self.load_record()
        if not status:
            print('load record 出問題')
            return

        self.result = True

    def get_face_grid(self) -> bool:
        self.face_grid = FaceGrid(self.image, self.image_name)
        self.face_grid.fetch()

        if self.face_grid.result:
            return True
        else:
            return False

    def load_roi_points(self, fg: FaceGrid) -> bool:
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

        return True

    def scale_dist_helper(self) -> bool:
        roi_size = self.roi_width * self.roi_height
        base = round(roi_size ** 0.05)

        if base == 0:
            base = 1

        self.scale_dist = list(range(0, base * (self.scale_range + 1), base))

        return True

    def output_handler(self) -> bool:
        if self.strict:
            status = self.strict_output(self.image)
        else:
            status = self.elastic_output(self.image)

        return status

    def get_roi(self, image: numpy.ndarray, scale_dist=10) -> numpy.ndarray:
        temp = image.copy()

        top_x = self.roi_top_x - scale_dist
        top_y = self.roi_top_y - scale_dist
        bot_x = self.roi_bot_x + scale_dist
        bot_y = self.roi_bot_y + scale_dist

        return temp[top_y:bot_y, top_x:bot_x]

    def strict_output(self, image: numpy.ndarray) -> bool:
        roi = self.get_roi(image, self.scale_dist[self.scale_range])
        if FBHelper.get_face_block(roi):
            cv2.imwrite(self.dest_path, roi)
            return True
        else:
            print('[WARN] - Failed to extract Face Grid - mode: strict')
            return False

    def elastic_output(self, image: numpy.ndarray) -> bool:
        status = False
        for scale_dist in reversed(self.scale_dist):
            try:
                roi = self.get_roi(image, scale_dist)

                if FBHelper.get_face_block(roi):
                    cv2.imwrite(self.dest_path, roi)
                    self.roi_image = roi
                    self.roi_image_w = self.roi_image.shape[0]
                    self.roi_image_h = self.roi_image.shape[1]
                    self.roi_image_size = self.roi_image_w * self.roi_image_h
                    # print('成功輸出至 - {}'.format(self.dest_path))
                    return True

            except Exception as e:
                # print('Crop Failed : {} - scale dist : {}'.format(self.image_name, scale_dist))
                print(LogTool.pp_exception(e))
                continue

        return status

    def load_record(self) -> bool:
        temp = [self.roi_width, self.roi_height, self.roi_size,
                self.roi_image_w, self.roi_image_h, self.roi_image_size]

        self.record.extend(temp)

        return True
