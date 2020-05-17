# coding: utf-8
import dlib
import numpy
from module.jf_ult.geom_tool import GeomTool


class LMKHelper:
    # landmark module
    predictor_path = 'module/models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    @staticmethod
    def get_landmarks(img: numpy.ndarray, face_block: dlib.rectangle) -> dict:
        result = dict()
        face_points = numpy.matrix([[p.x, p.y] for p in LMKHelper.predictor(img, face_block).parts()])
        for index, point in enumerate(face_points):
            pos = (point[0, 0], point[0, 1])
            # lm_id = index + 1
            lm_id = str(index + 1)
            result[lm_id] = pos

        return result

    @staticmethod  # support point function
    def add_eye_mid(landmarks: dict) -> dict:
        # Get the middle of right eye
        right_eye_ql13 = (landmarks['42'], landmarks['39'])
        right_eye_ql24 = (landmarks['38'], landmarks['41'])
        right_eye_mid = GeomTool.get_line_intersect(right_eye_ql13, right_eye_ql24)
        landmarks['s1'] = right_eye_mid

        # Get the middle of left eye
        left_eye_ql13 = (landmarks['45'], landmarks['48'])
        left_eye_ql24 = (landmarks['44'], landmarks['47'])
        left_eye_mid = GeomTool.get_line_intersect(left_eye_ql13, left_eye_ql24)
        landmarks['s2'] = left_eye_mid

        return landmarks

    @staticmethod
    def add_lip_mid(landmarks: dict) -> dict:
        lip_vl = (landmarks['52'], landmarks['58'])
        lip_hl = (landmarks['49'], landmarks['55'])
        lip_mid = GeomTool.get_line_intersect(lip_vl, lip_hl)
        landmarks['s3'] = lip_mid

        return landmarks
