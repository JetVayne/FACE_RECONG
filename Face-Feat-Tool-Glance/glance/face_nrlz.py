# coding: utf-8
import os
import cv2
from glance.face_yrp import FaceYRP
from glance.face_rotater import FaceRotator
from glance.fg_wizard import FGWizard
import numpy


class FaceNormalizor:
    def __init__(self, image: str, image_name=None, dest_path=None):
        self.YRP_threshold = 10
        self.image_path = None
        self.image = None

        if isinstance(image, numpy.ndarray):
            self.image = image

        self.image_name = image_name
        self.dest_path = dest_path




        self.result = False

        # fg_width, fg_height, fg_size, fg_image, fg_image_w, fg_image_h, fg_image_size
        self.record = list()

        self.image_afr = None

        self.fg_width = 0
        self.fg_height = 0
        self.fg_size = 0

        self.fg_image = None
        self.fg_image_w = None
        self.fg_image_h = None
        self.fg_image_size = None

    def launch(self):
        if self.image is None:
            # load image
            status = self.load_image()
            if not status:
                return

        # check YRP
        status = self.check_YRP()
        if not status:
            return

        # get rotated image
        status = self.rotate_image()
        if not status:
            return

        # get the face grid by Face Grid Wizard
        status = self.acquire_face_grid()
        if not status:
            return

        self.result = True

    def load_image(self) -> bool:
        if os.path.exists(self.image_path):
            # self.image = ImgProcTool.img_read(self.image_path)
            self.image = cv2.imread(self.image_path, 1)
            return True
        else:
            print('[WARN] - IMG is not exist - {}'.format(self.image_path))
            return False

    def check_YRP(self) -> bool:
        face_yrp = FaceYRP(self.image, image_name=self.image_name)
        face_yrp.launch()

        if face_yrp.result:
            if - self.YRP_threshold < face_yrp.pitch < self.YRP_threshold and - self.YRP_threshold < face_yrp.yaw < self.YRP_threshold:
                return True
        else:
            print('[INFO] - Failed to get Face YRP - {}'.format(self.image_path))
            return False

    def rotate_image(self) -> bool:
        face_rt = FaceRotator(self.image, self.image_name)
        face_rt.launch()

        if face_rt.result:
            self.image_afr = face_rt.rt_image
            return True
        else:
            print('[INFO] - Failed to rotated - {}'.format(self.image_path))
            return False

    def acquire_face_grid(self) -> bool:
        fg_wizard = FGWizard(self.image_afr, self.image_name, self.dest_path)
        fg_wizard.launch()

        if fg_wizard.result:
            self.load_record(fg_wizard)
            return True

        else:
            print('[INFO] - Face Grid Wizard failed to extract face grid - {}'.format(self.image_path))
            return False

    def load_record(self, fg_wizard: FGWizard):
        self.fg_width = fg_wizard.roi_width
        self.fg_height = fg_wizard.roi_height
        self.fg_size = fg_wizard.roi_size

        self.fg_image = fg_wizard.roi_image

        self.fg_image_w = fg_wizard.roi_image_w
        self.fg_image_h = fg_wizard.roi_image_h
        self.fg_image_size = fg_wizard.roi_image_size

        temp = [self.fg_width, self.fg_height, self.fg_size,
                self.fg_image_w, self.fg_image_h, self.fg_image_size]
        self.record.extend(temp)
