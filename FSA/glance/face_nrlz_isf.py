# coding: utf-8
import os

import cv2
import numpy

from glance.face_rotater import FaceRotator
from glance.face_yrp import FaceYRP
from glance.fg_wizard import FGWizard
from glance.jf_ult.isf_wizard import ISFWizard
from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper


class FaceNrlzISF:
    def __init__(self, image, isf_model, image_name=None, dest_path=None):
        self.YRP_threshold = 10
        self.isf_model = isf_model
        self.isf_model = self.isf_model
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

        self.gender = None
        self.age = None

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

        print('[LOCAL INFO] - YRP 測完 - {}'.format(self.image_path))

        # get rotated image
        status = self.rotate_image()
        if not status:
            return

        print('[LOCAL INFO] - 旋轉完成 - {}'.format(self.image_path))

        # get the face grid by Face Grid Wizard
        status = self.acquire_face_grid()
        if not status:
            return

        print('[LOCAL INFO] - FG 抓完 - {}'.format(self.image_path))

        self.result = True

    def load_image(self) -> bool:
        if os.path.exists(self.image_path):
            # self.image = ImgProcTool.img_read(self.image_path)
            self.image = cv2.imread(self.image_path, 1)
            return True
        else:
            print('[LOCAL WARN] - IMG is not exist - {}'.format(self.image_path))
            return False

    def check_YRP(self) -> bool:
        face_yrp = FaceYRP(self.image, image_name=self.image_name)
        face_yrp.launch()

        if face_yrp.result:
            if - self.YRP_threshold < face_yrp.pitch < self.YRP_threshold and - self.YRP_threshold < face_yrp.yaw < self.YRP_threshold:
                return True
        else:
            print('[LOCAL INFO] - Failed to get Face YRP - {}'.format(self.image_path))
            return False

    def check_dlib_acc(self, image: numpy.ndarray) -> tuple:
        result = False
        isf_wizard = ISFWizard(self.isf_model, image)
        isf_wizard.launch()
        if not isf_wizard:
            return result, isf_wizard
        isf_lmks = isf_wizard.isf_lmks

        fb_temp = FBHelper.get_face_block(image)
        if not fb_temp:
            print('[LOCAL] - {} - 找不到 Face Block'.format(__name__))
            return result, isf_wizard
        dlib_lmks = LMKHelper.get_landmarks(image, fb_temp)

        dlib_acc = LMKHelper.get_lmks_acc(dlib_lmks, isf_lmks)

        if dlib_acc < 0.05:
            result = True

        return result, isf_wizard

    def rotate_image(self) -> bool:
        # dlib_acc, isf_wizard = self.check_dlib_acc(self.image)
        dlib_acc = True
        if not dlib_acc:
            print('[LOCAL] - {} - DLIB 誤差超過 5 % - {}'.format(__name__, self.image_name))
            return False

        face_rt = FaceRotator(self.image, self.image_name)
        face_rt.launch()

        if face_rt.result:
            self.image_afr = face_rt.rt_image
            # dlib_acc, isf_wizard = self.check_dlib_acc(self.image_afr)
            if dlib_acc:
                # self.gender = isf_wizard.gender
                # self.age = isf_wizard.age
                return True
            else:
                print('[LOCAL] - {} - 旋轉後 DLIB 誤差超過 5 % - {}'.format(__name__, self.image_name))
                return False
        else:
            print('[LOCAL INFO] - Failed to rotated - {}'.format(self.image_path))
            return False

    def acquire_face_grid(self) -> bool:
        fg_wizard = FGWizard(self.image_afr, self.image_name, self.dest_path)
        fg_wizard.launch()

        if fg_wizard.result:
            self.load_record(fg_wizard)
            return True

        else:
            print('[LOCAL INFO] - Face Grid Wizard failed to extract face grid - {}'.format(self.image_path))
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

        self.record.append(self.gender)
        self.record.append(self.age)
        self.record.extend(temp)
