# coding: utf-8
import numpy


class ISFWizard:
    def __init__(self, isf_model, image: numpy.ndarray):
        self.model = isf_model
        self.image = image

        self.result = False
        self.msg = ''

        self.gender = None
        self.age = None
        self.isf_lmks = dict()

    @staticmethod
    def get_isf_lmks(coords) -> dict:
        result = dict()
        c_key = ['ppr', 'ppl', '31', '49', '55']
        for key, c in zip(c_key, coords):
            result[key] = (c[0], c[1])

        return result

    @staticmethod
    def get_gender(face) -> str:
        if face.gender:
            return 'm'
        else:
            return 'f'

    def launch(self):
        detect_faces = self.model.get(self.image)
        if len(detect_faces) == 1:
            face = detect_faces[0]
            self.age = face.age
            self.gender = ISFWizard.get_gender(face)
            self.isf_lmks = ISFWizard.get_isf_lmks(face.landmark)
            self.result = True

        else:
            print('[LOCAL] - {} - Insight Face 找到的臉數 != 1'.format(__name__))
            self.msg = '人臉數 != 1'
