# coding:
import numpy
from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper
from glance.face_yrp import FaceYRP

class YrpValidator:
    def __init__(self, image: numpy.ndarray):
        self.YRP_threshold = 20
        self.image = image
        self.result = False
        self.msg = ''

        self.face_block = None
        self.landmarks = dict()
        self.scan()

    def scan(self):
        face_yrp = FaceYRP(self.image)
        face_yrp.launch()

        if face_yrp.result:
            print('[LOCAL] - {} - 趨近正臉'.format(__name__))
            print(face_yrp.pitch, face_yrp.yaw)
            if - self.YRP_threshold < face_yrp.pitch < self.YRP_threshold and - self.YRP_threshold < face_yrp.yaw < self.YRP_threshold:
                self.result = True

            else:
                self.msg = '照片中人物的臉部擺動幅度過大，請盡量保持正臉!'
                self.result = False

        else:
            print('[LOCAL] - {} - 非正臉'.format(__name__))
            self.msg = '無法評估照片中人物的臉部角度'
            self.result = False
