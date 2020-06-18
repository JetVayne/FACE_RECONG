# coding:
import numpy
from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper


class FaceValidator:
    def __init__(self, image: numpy.ndarray):
        self.image = image
        self.result = False
        self.msg = ''

        self.face_block = None
        self.landmarks = dict()
        self.scan()

    def scan(self):
        status = self.check_face_count()
        if not status:
            return

        print('[LOCAL] - {} - 找的到臉'.format(__name__))

        status = self.can_find_landmarks()
        if status:
            print('[LOCAL] - {} - 找的到 landmarks'.format(__name__))
            self.result = True

    def check_face_count(self) -> bool:
        self.face_block = FBHelper.get_face_block(self.image)

        if self.face_block:
            return True

        else:
            self.msg = '照片偵測不到人臉，或照片中的人物超過 1 位'
            print('[LOCAL] - {} - {}'.format(__name__, self.msg))
            return False

    def can_find_landmarks(self) -> bool:
        self.landmarks = LMKHelper.get_landmarks(self.image, self.face_block)
        if self.landmarks:
            return True

        else:
            self.msg = '無法獲取照片中人臉的特徵'
            print('[LOCAL] - {} - {}'.format(__name__, self.msg))
            return False
