# coding: utf-8
import numpy

from glance.face_scale.ed import ED
from glance.face_scale.nlj import NLJ

from glance.face_feat.ratio.cjwr import CJWR
from glance.face_feat.ratio.fmwr import FMWR
from glance.face_feat.ratio.esr import ESR
from glance.face_feat.ratio.ebhr import EBHR
from glance.face_feat.ratio.tr import TR
from glance.face_feat.ratio.itr import ITR
from glance.face_feat.ratio.fcr import FCR
from glance.face_feat.ratio.ucr import UCR

from glance.face_feat.area.ta import TA
from glance.face_feat.area.ita import ITA
from glance.face_feat.area.fca import FCA
from glance.face_feat.area.uca import UCA
from glance.face_feat.area.ftza import FTZA
from glance.face_feat.area.fpya import FPYA
from glance.face_feat.area.cfa import CFA

from glance.face_feat.index.lfi import LFI
from glance.face_feat.index.bci import BCI
from glance.face_feat.index.cfi import CFI

from glance.jf_ult.fb_helper import FBHelper
from glance.jf_ult.lmk_helper import LMKHelper


class FaceFeatHelper:
    def __init__(self, image: numpy.ndarray, image_name='', fg_size=None, mode='normal', dest_folder_path=None):
        self.image = image.copy()
        self.image_name = image_name
        if '.jpg' in self.image_name:
            self.image_name = self.image_name.replace('.jpg', '')

        self.fg_size = fg_size
        self.mode = mode
        self.dest_folder = dest_folder_path

        self.result = None

        self.face_block = None
        self.landmarks = dict()

        self.scale_feat_list = ['ED', 'NLJ']

        self.bmi_feat_list = ['CJWR', ' CJWR2',
                              'FMWR', 'FMWR2',
                              'ESR', 'ESR2', 'ESR3', 'ESR4',
                              'EBHR', 'EBHR2', 'EBHR3', 'EBHR4', 'EBHR5',
                              'TR', 'TR2', 'TR3', 'TR4',
                              'ITR', 'ITR2', 'ITR3', 'ITR4',
                              'FCR', 'FCR2', 'FCR3', 'FCR4', 'FCR5',
                              'UCR', 'UCR2', 'UCR3', 'UCR4', 'UCR5',
                              'TA', 'TA2', 'TA3', 'TA4',
                              'ITA', 'ITA2', 'ITA3', 'ITA4',
                              'FCA', 'FCA2', 'FCA3', 'FCA4',
                              'UCA', 'UCA2', 'UCA3', 'UCA4',
                              'FTZA', 'FTZA2',
                              'FPYA', 'FPYA2',
                              'CFA', 'CFA2',
                              'LFI', 'LFI2',
                              'BCI', 'BCI2',
                              'CFI', 'CFI2', 'CFI3', 'CFI4', 'CFI5', 'CFI6'
                              ]

        self.all_feat_list = self.scale_feat_list + self.bmi_feat_list

        # output record
        self.features = dict()
        self.features_vlist = list()

        self.features_scale = dict()
        self.features_scale_vlist = list()

        self.features_all = dict()
        self.features_all_vlist = list()

        # Scale Feat
        self.ED = None
        self.NLJ = None

        # Ratio Feat
        self.CJWR = None
        self.CJWR2 = None

        self.FMWR = None
        self.FMWR2 = None

        self.ESR = None
        self.ESR2 = None
        self.ESR3 = None
        self.ESR4 = None

        self.EBHR = None
        self.EBHR2 = None
        self.EBHR3 = None
        self.EBHR4 = None
        self.EBHR5 = None

        self.TR = None
        self.TR2 = None
        self.TR3 = None
        self.TR4 = None

        self.ITR = None
        self.ITR2 = None
        self.ITR3 = None
        self.ITR4 = None

        self.FCR = None
        self.FCR2 = None
        self.FCR3 = None
        self.FCR4 = None
        self.FCR5 = None

        self.UCR = None
        self.UCR2 = None
        self.UCR3 = None
        self.UCR4 = None
        self.UCR5 = None

        # Area Feat
        self.TA = None
        self.TA2 = None
        self.TA3 = None
        self.TA4 = None

        self.ITA = None
        self.ITA2 = None
        self.ITA3 = None
        self.ITA4 = None

        self.FCA = None
        self.FCA2 = None
        self.FCA3 = None
        self.FCA4 = None

        self.UCA = None
        self.UCA2 = None
        self.UCA3 = None
        self.UCA4 = None

        self.FTZA = None
        self.FTZA2 = None

        self.FPYA = None
        self.FPYA2 = None

        self.CFA = None
        self.CFA2 = None

        # Index Feat
        self.LFI = None
        self.LFI2 = None

        self.BCI = None
        self.BCI2 = None

        self.CFI = None
        self.CFI2 = None
        self.CFI3 = None
        self.CFI4 = None
        self.CFI5 = None
        self.CFI6 = None

    def load_landmarks(self):
        self.landmarks = LMKHelper.get_landmarks(self.image, self.face_block)
        # self.landmarks = LMKHelper.add_eye_mid(self.landmarks)
        # self.landmarks = LMKHelper.add_lip_mid(self.landmarks)

    def launch(self):
        self.face_block = FBHelper.get_face_block(self.image)
        if self.face_block is None:
            print('[WARN] - Failed to fetch face block, img: {}'.format(self.image_name))
            self.result = False
            return

        # load landmarks
        self.load_landmarks()

        # load scale-feat (for observed)
        self.load_scale_feat()

        # load ratio-feat
        self.load_ratio_feat()

        # load area-feat
        self.load_area_feat()

        # load index-feat
        self.load_index_feat()

        # gen record dicts
        self.gen_record()

        self.result = True

    def gen_record(self):
        # normal
        for f in self.bmi_feat_list:
            val = eval('self.{}'.format(f))
            self.features[f] = val
            self.features_vlist.append(val)

        # scale
        for f in self.scale_feat_list:
            val = eval('self.{}'.format(f))
            self.features_scale[f] = val
            self.features_scale_vlist.append(val)

        # all
        for f in self.all_feat_list:
            val = eval('self.{}'.format(f))
            self.features_all[f] = val
            self.features_all_vlist.append(val)

    def load_scale_feat(self):
        self.load_ED()
        self.load_NLJ()

    def load_ratio_feat(self):
        self.load_CJWR()
        self.load_FMWR()

        self.load_ESR()
        self.load_EBHR()

        self.load_TR()
        self.load_ITR()

        self.load_FCR()
        self.load_UCR()

    def load_area_feat(self):
        self.load_FCA()
        self.load_UCA()

        self.load_TA()
        self.load_ITA()

        self.load_FTZA()
        self.load_FPYA()
        self.load_CFA()

    def load_index_feat(self):
        self.load_LFI()
        self.load_BCI()
        self.load_CFI()

    # ------------------------------------- SCALE TYPE -------------------------------------
    def load_ED(self):
        ed = ED(self.landmarks, self.image)
        # ed.show()
        if self.mode != 'normal':
            ed.show('{}/{}-ed.jpg'.format(self.dest_folder, self.image_name))

        self.ED = ed.val
        # print('ED: {}'.format(self.ED))

    def load_NLJ(self):
        nlj = NLJ(self.landmarks, self.image)
        # nlj.show()
        if self.mode != 'normal':
            nlj.show('{}/{}-nlj.jpg'.format(self.dest_folder, self.image_name))

        self.NLJ = nlj.val
        # print('NLJ: {}'.format(self.NLJ))

    # ------------------------------------- RATIO TYPE -------------------------------------
    def load_CJWR(self):
        cjwr = CJWR(self.landmarks, self.image)
        # cjwr.show()
        if self.mode != 'normal':
            cjwr.show('{}/{}-cjwr.jpg'.format(self.dest_folder, self.image_name))

        self.CJWR = cjwr.val
        self.CJWR2 = cjwr.val_2

        # print('CJWR: {}'.format(self.CJWR))

    def load_FMWR(self):
        fmwr = FMWR(self.landmarks, self.image)
        # fwmr.show()
        if self.mode != 'normal':
            fmwr.show('{}/{}-fmwr.jpg'.format(self.dest_folder, self.image_name))

        self.FMWR = fmwr.val
        self.FMWR2 = fmwr.val_2

        # print('FMWR: {}'.format(self.FMWR))

    def load_ESR(self):
        esr = ESR(self.landmarks, self.image)
        # esr.show()
        if self.mode != 'normal':
            esr.show('{}/{}-esr.jpg'.format(self.dest_folder, self.image_name))

        self.ESR = esr.val
        self.ESR2 = esr.val_2
        self.ESR3 = esr.val_3
        self.ESR4 = esr.val_4

        # print('ESR: {}'.format(self.ESR))

    def load_EBHR(self):
        ebhr = EBHR(self.landmarks, self.image)
        # ebhr.show()
        if self.mode != 'normal':
            ebhr.show('{}/{}-ebhr.jpg'.format(self.dest_folder, self.image_name))

        self.EBHR = ebhr.val
        self.EBHR2 = ebhr.val_2
        self.EBHR3 = ebhr.val_3
        self.EBHR4 = ebhr.val_4
        self.EBHR5 = ebhr.val_5

        # print('EBHR: {}'.format(self.EBHR))

    def load_TR(self):
        tr = TR(self.landmarks, self.image)
        # tr.show()
        if self.mode != 'normal':
            tr.show('{}/{}-tr.jpg'.format(self.dest_folder, self.image_name))

        self.TR = tr.val
        self.TR2 = tr.val_2
        self.TR3 = tr.val_3
        self.TR4 = tr.val_4

        # print('TR: {}'.format(self.TR))

    def load_ITR(self):
        itr = ITR(self.landmarks, self.image)
        # itr.show()
        if self.mode != 'normal':
            itr.show('{}/{}-itr.jpg'.format(self.dest_folder, self.image_name))

        self.ITR = itr.val
        self.ITR2 = itr.val_2
        self.ITR3 = itr.val_3
        self.ITR4 = itr.val_4

        # print('ITR: {}'.format(self.ITR))

    def load_FCR(self):
        fcr = FCR(self.landmarks, self.image)
        # fcr.show()
        if self.mode != 'normal':
            fcr.show('{}/{}-fcr.jpg'.format(self.dest_folder, self.image_name))

        self.FCR = fcr.val
        self.FCR2 = fcr.val_2
        self.FCR3 = fcr.val_3
        self.FCR4 = fcr.val_4

        # print('FCR: {}'.format(self.FCR))

    def load_UCR(self):
        ucr = UCR(self.landmarks, self.image)
        # ucr.show()
        if self.mode != 'normal':
            ucr.show('{}/{}-ucr.jpg'.format(self.dest_folder, self.image_name))

        self.UCR = ucr.val
        self.UCR2 = ucr.val_2
        self.UCR3 = ucr.val_3
        self.UCR4 = ucr.val_4

        # print('UCR: {}'.format(self.UCR))

    # ------------------------------------- AREA TYPE -------------------------------------
    def load_TA(self):
        ta = TA(self.landmarks, self.image)
        # ta.show()
        if self.mode != 'normal':
            ta.show('{}/{}-ta.jpg'.format(self.dest_folder, self.image_name))

        self.TA = ta.val
        self.TA2 = ta.val_2
        self.TA3 = ta.val_3
        self.TA4 = ta.val_4

        # print('TA: {}'.format(self.TA))

    def load_ITA(self):
        ita = ITA(self.landmarks, self.image)
        # ita.show()
        if self.mode != 'normal':
            ita.show('{}/{}-ita.jpg'.format(self.dest_folder, self.image_name))

        self.ITA = ita.val
        self.ITA2 = ita.val_2
        self.ITA3 = ita.val_3
        self.ITA4 = ita.val_4

        # print('ITA: {}'.format(self.ITA))

    def load_FCA(self):
        fca = FCA(self.landmarks, self.image)
        # fca.show()
        if self.mode != 'normal':
            fca.show('{}/{}-fca.jpg'.format(self.dest_folder, self.image_name))

        self.FCA = fca.val
        self.FCA2 = fca.val_2
        self.FCA3 = fca.val_3
        self.FCA4 = fca.val_4

        # print('FCA: {}'.format(self.FCA))

    def load_UCA(self):
        uca = UCA(self.landmarks, self.image)
        # uca.show()
        if self.mode != 'normal':
            uca.show('{}/{}-uca.jpg'.format(self.dest_folder, self.image_name))

        self.UCA = uca.val
        self.UCA2 = uca.val_2
        self.UCA3 = uca.val_3
        self.UCA4 = uca.val_4

        # print('UCA: {}'.format(self.UCA))

    def load_FTZA(self):
        ftza = FTZA(self.landmarks, self.image)
        # ftza.show()
        if self.mode != 'normal':
            ftza.show('{}/{}-ftza.jpg'.format(self.dest_folder, self.image_name))

        self.FTZA = ftza.val
        self.FTZA2 = ftza.val_2

        # print('FTZA: {}'.format(self.FTZA))

    def load_FPYA(self):
        fpya = FPYA(self.landmarks, self.image)
        # fpya.show()
        if self.mode != 'normal':
            fpya.show('{}/{}-fpya.jpg'.format(self.dest_folder, self.image_name))

        self.FPYA = fpya.val
        self.FPYA2 = fpya.val_2

        # print('FPYA: {}'.format(self.FPYA))

    def load_CFA(self):
        cfa = CFA(self.landmarks, self.image)
        # cfa.show()
        if self.mode != 'normal':
            cfa.show('{}/{}-cfa.jpg'.format(self.dest_folder, self.image_name))

        self.CFA = cfa.val
        self.CFA2 = cfa.val_2

        # print('CFA: {}'.format(self.CFA))

    # ------------------------------------- INDEX TYPE -------------------------------------
    def load_LFI(self):
        lfi = LFI(self.landmarks, self.image)
        # lfi.show()
        if self.mode != 'normal':
            lfi.show('{}/{}-lfi.jpg'.format(self.dest_folder, self.image_name))

        self.LFI = lfi.val
        self.LFI2 = lfi.val_2

        # print('LFI: {}'.format(self.LFI))

    def load_BCI(self):
        bci = BCI(self.landmarks, self.image)
        # bci.show()
        if self.mode != 'normal':
            bci.show('{}/{}-bci.jpg'.format(self.dest_folder, self.image_name))

        self.BCI = bci.val
        self.BCI2 = bci.val_2

        # print('BCI: {}'.format(self.BCI))

    def load_CFI(self):
        cfi = CFI(self.landmarks, self.image)
        # cfi.show()
        if self.mode != 'normal':
            cfi.show('{}/{}-cfi.jpg'.format(self.dest_folder, self.image_name))

        self.CFI = cfi.val
        self.CFI2 = cfi.val_2
        self.CFI3 = cfi.val_3
        self.CFI4 = cfi.val_4
        self.CFI5 = cfi.val_5
        self.CFI6 = cfi.val_6

        # print('CFI: {}'.format(self.CFI))
