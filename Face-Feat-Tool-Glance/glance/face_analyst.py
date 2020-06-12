# coding: utf-8
import os

import cv2

from glance.face_feat_helper import FaceFeatHelper
from glance.face_rotater import FaceRotator
from glance.face_yrp import FaceYRP
from glance.fg_wizard import FGWizard
from glance.gar_wizard import GarWizard
from glance.jf_ult.log_tool import LogTool


class FaceAnalyst:
    def __init__(self, img_path: str, img_name='', crop_save_path='saved_facegrid', gen_ff_img=False,
                 ff_save_path='saved_ff_img'):
        # input
        self.image_path = img_path
        if img_name == '':
            self.image_name = self.image_path.split('/')[-1]
        else:
            self.image_name = img_name

        self.img_os_info = 'IMG-PATH: {}  IMG-NAME: {}'.format(self.image_path, self.image_name)
        self.gen_ff_img = gen_ff_img

        self.face_grid_dir = crop_save_path
        if self.face_grid_dir == 'saved_facegrid':
            if not os.path.exists(self.face_grid_dir):
                os.makedirs(self.face_grid_dir)

        self.ff_dir = ff_save_path
        if self.ff_dir == 'saved_ff_img':
            if not os.path.exists(self.ff_dir):
                os.makedirs(self.ff_dir)

        self.face_grid_path = '{}/{}'.format(self.face_grid_dir, self.image_name)
        self.result = False

        # face info from Deep-Face
        self.image = None
        self.gender = None
        self.age = None
        self.race = None
        self.emotion = None

        # head pose from face_yrp.py
        self.roll = None
        self.yaw = None
        self.pitch = None

        # face grid meta
        self.fg_width = None
        self.fg_height = None
        self.fg_size = None

        # face features
        self.ff_helper = None
        self.features = dict()
        self.features_vlist = list()
        self.features_all = dict()  # contain scale feat
        self.features_all_vlist = list()  # contain scale feat value

        self.data = dict()

    def load_image(self):
        if os.path.exists(self.image_path):
            try:
                self.image = cv2.imread(self.image_path)
                return True
            except Exception as e:
                msg = 'Failed to read image by OpenCV. {}'.format(self.img_os_info)
                msg += LogTool.pp_exception(e)
                print(msg)
                return False
        else:
            msg = 'Image Path is not exists. {}'.format(self.img_os_info)
            print(msg)
            return False

    def get_face_info(self):
        deep_face_info = GarWizard.get_face_dict(self.image_path)
        if deep_face_info:
            gender = deep_face_info.get('gender')
            if gender == 'Man':
                self.gender = 'Male'
            else:
                self.gender = 'Female'

            self.age = deep_face_info.get('age')
            self.race = deep_face_info.get('dominant_race')
            self.emotion = deep_face_info.get('dominant_emotion')

            return True

        else:
            return False

    def get_yrp(self):
        face_yrp = FaceYRP(self.image)
        face_yrp.launch()

        if face_yrp.result:
            self.yaw = face_yrp.yaw
            self.roll = face_yrp.roll
            self.pitch = face_yrp.pitch
            return True

        else:
            return False

    def get_face_grid(self):
        face_rt = FaceRotator(self.image, self.image_name)
        face_rt.launch()
        if not face_rt.result:
            return False
        rotated_image = face_rt.rt_image

        fg_wizard = FGWizard(rotated_image, img_name=self.image_name, dest_path=self.face_grid_path)
        fg_wizard.launch()
        if not fg_wizard.result:
            return False

        self.image = fg_wizard.roi_image
        self.fg_width = fg_wizard.roi_width
        self.fg_height = fg_wizard.roi_height
        self.fg_size = fg_wizard.roi_size

        return True

    def get_face_feature(self):
        mode = 'normal'
        dest_dir = None
        if self.gen_ff_img:
            mode = 'o'
            dest_dir = self.ff_dir

        self.ff_helper = FaceFeatHelper(self.image, self.image_name, mode=mode, dest_folder_path=dest_dir)
        self.ff_helper.launch()

        if self.ff_helper.result:
            self.features = self.ff_helper.features
            self.features_vlist = self.ff_helper.features_vlist

            self.features_all = self.ff_helper.features_all
            self.features_all_vlist = self.ff_helper.features_all_vlist

            return True

        else:
            return False

    def load_data(self):
        self.data['gender'] = self.gender
        self.data['age'] = self.age
        self.data['race'] = self.race
        self.data['roll'] = self.roll
        self.data['yaw'] = self.yaw
        self.data['pitch'] = self.pitch
        self.data['fg_width'] = self.fg_width
        self.data['fg_height'] = self.fg_height
        self.data['fg_size'] = self.fg_size
        self.data['feat'] = self.features_all
        # self.data['feat-all'] = self.features_all

    def analyze(self):
        status = self.load_image()
        if not status:
            return

        # status = self.get_face_info()
        # if not status:
        #     msg = 'Failed to get face info by Deep-Face. {}'.format(self.img_os_info)
        #     print(msg)
        #     return

        status = self.get_yrp()
        if not status:
            msg = 'Failed to get head pose by Face-YRP. {}'.format(self.img_os_info)
            print(msg)
            return

        status = self.get_face_grid()
        if not status:
            msg = 'Failed to get Face-Grid. {}'.format(self.img_os_info)
            print(msg)
            return

        status = self.get_face_feature()
        if not status:
            msg = 'Failed to get Face-Features. {}'.format(self.img_os_info)
            print(msg)
            return

        self.load_data()
        self.result = True
