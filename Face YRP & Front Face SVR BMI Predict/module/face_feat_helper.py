# coding: utf-8
import numpy

from module.face_feat.cfr import CFR
from module.face_feat.cjwr import CJWR
from module.face_feat.es import ES
from module.face_feat.fwr import FWR
from module.face_feat.meh import MEH
from module.face_feat.par import PAR
from module.face_feat.whr import WHR
from module.jf_ult.fb_helper import FBHelper
from module.jf_ult.lmk_helper import LMKHelper


class FaceFeatHelper:
    def __init__(self, image: numpy.ndarray, image_name='', mode='normal', dest_folder_path=None):
        self.image = image.copy()
        self.image_name = image_name
        self.mode = mode
        self.dest_folder = dest_folder_path

        self.result = None

        self.face_block = None
        self.landmarks = dict()

        # Face Feature
        self.CJWR = None  # cheekbone to jaw width
        self.WHR = None  # jaw width to upper facial height ratio
        self.PAR = None  # perimeter to area ratio
        self.ES = None  # eye size
        self.FWR = None  # face width to lower face height ratio
        self.MEH = None  # mean of eyebrow height
        self.CFR = None  # cheek fat ratio

        self.features = list()

    def launch(self):
        self.face_block = FBHelper.get_face_block(self.image)
        if self.face_block is None:
            print('[WARN] - Failed to fetch face block, img: {}'.format(self.image_name))
            self.result = False
            return
        ## load landmarks
        self.load_landmarks()

        ## load face feature
        self.load_CJWR()
        self.load_WHR()
        self.load_PAR()
        self.load_ES()
        self.load_FWR()
        self.load_MEH()
        self.load_CFR()

        self.result = True
        self.load_feature()

    def load_feature(self):
        self.features.append(self.CJWR)
        self.features.append(self.WHR)
        self.features.append(self.PAR)
        self.features.append(self.ES)
        self.features.append(self.FWR)
        self.features.append(self.MEH)
        self.features.append(self.CFR)

    def load_landmarks(self):
        self.landmarks = LMKHelper.get_landmarks(self.image, self.face_block)
        self.landmarks = LMKHelper.add_eye_mid(self.landmarks)
        self.landmarks = LMKHelper.add_lip_mid(self.landmarks)

    def load_CJWR(self):
        cjwr = CJWR(self.landmarks, self.image)
        # cjwr.show()
        if self.mode != 'normal':
            cjwr.show('{}/{}-cjwr.jpg'.format(self.dest_folder, self.image_name))

        self.CJWR = round(cjwr.val, 2)
        print('CJWR: {}'.format(self.CJWR))

    def load_WHR(self):
        whr = WHR(self.landmarks, self.image)
        # whr.show()
        if self.mode != 'normal':
            whr.show('{}/{}-whr.jpg'.format(self.dest_folder, self.image_name))

        self.WHR = round(whr.val, 2)
        print('WWR: {}'.format(self.WHR))

    def load_PAR(self):
        par = PAR(self.landmarks, self.image)
        # par.show()
        if self.mode != 'normal':
            par.show('{}/{}-par.jpg'.format(self.dest_folder, self.image_name))

        self.PAR = round(par.val, 2)
        print('PAR: {}'.format(self.PAR))

    def load_ES(self):
        es = ES(self.landmarks, self.image)
        # es.show()
        if self.mode != 'normal':
            es.show('{}/{}-es.jpg'.format(self.dest_folder, self.image_name))

        self.ES = round(es.val, 2)
        print('ES: {}'.format(self.ES))

    def load_FWR(self):
        fwr = FWR(self.landmarks, self.image)
        # fwr.show()
        if self.mode != 'normal':
            fwr.show('{}/{}-fwr.jpg'.format(self.dest_folder, self.image_name))

        self.FWR = round(fwr.val, 2)
        print('FWR: {}'.format(self.FWR))

    def load_MEH(self):
        meh = MEH(self.landmarks, self.image)
        # meh.show()
        if self.mode != 'normal':
            meh.show('{}/{}-meh.jpg'.format(self.dest_folder, self.image_name))

        self.MEH = round(meh.val, 2)
        print('MEH: {}'.format(self.MEH))

    def load_CFR(self):
        cfr = CFR(self.landmarks, self.image)
        # cfr.show()
        if self.mode != 'normal':
            cfr.show('{}/{}-cfr.jpg'.format(self.dest_folder, self.image_name))

        self.CFR = round(cfr.val, 2)
        print('CFR: {}'.format(self.CFR))
