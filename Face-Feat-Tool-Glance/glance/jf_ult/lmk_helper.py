# coding: utf-8
import pathlib

import dlib
import numpy

from glance.jf_ult.geom_tool import GeomTool


class LMKHelper:
    # landmark glance
    predictor_path = '{}/models/shape_predictor_68_face_landmarks.dat'.format(pathlib.Path(__file__).parent.parent)
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

    @staticmethod
    def get_face_scale(landmarks: dict) -> dict:
        result = dict()
        # horizontal scale
        result['hrz'] = GeomTool.get_pt_dist(landmarks['40'], landmarks['43'])

        # vertical scale
        vrt_top = GeomTool.get_pt_dist(landmarks['34'], landmarks['58'])
        vrt_bot = GeomTool.get_pt_dist(landmarks['58'], landmarks['9'])
        vrt_sum = vrt_top + vrt_bot
        vrt_mean = (vrt_top + vrt_bot) / 2
        result['vrt'] = vrt_sum
        result['vrt2'] = vrt_mean
        result['vrt3'] = vrt_top
        result['vrt4'] = vrt_bot

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

    @staticmethod
    def get_reb_top(landmarks: dict) -> tuple:
        positions = [str(i) for i in range(18, 23, 1)]
        temp = dict()
        for p in positions:
            temp[p] = landmarks[p]

        sorted_lmks = sorted(temp.items(), key=lambda x: x[1][1])

        right_eyebrow_top = sorted_lmks[0][1]
        return tuple(right_eyebrow_top)

    @staticmethod
    def get_leb_top(landmarks: dict) -> tuple:
        positions = [str(i) for i in range(23, 28, 1)]
        temp = dict()
        for p in positions:
            temp[p] = landmarks[p]

        sorted_lmks = sorted(temp.items(), key=lambda x: x[1][1])

        left_eyebrow_top = sorted_lmks[0][1]
        return tuple(left_eyebrow_top)

    @staticmethod
    def get_edge_points(landmarks: dict, top_line: tuple) -> (tuple, tuple):
        # sorted_lmks = sorted(landmarks.items(), key=lambda x: x[1][0])
        sorted_lmks = sorted(landmarks.items(), key=lambda x: GeomTool.get_prj_point(x[1], top_line)[0])
        return sorted_lmks[0][1], sorted_lmks[-1][1]

    @staticmethod
    def get_fg_top_left(cheek_right: tuple, top_line: tuple):
        return GeomTool.get_prj_point(cheek_right, top_line)

    @staticmethod
    def get_fg_top_right(cheek_left: tuple, top_line: tuple):
        return GeomTool.get_prj_point(cheek_left, top_line)

    @staticmethod
    def get_fg_bot_left(jaw_tip: tuple, left_line: tuple):
        return GeomTool.get_prj_point(jaw_tip, left_line)

    @staticmethod
    def get_fg_bot_right(jaw_tip: tuple, right_line: tuple):
        return GeomTool.get_prj_point(jaw_tip, right_line)

    @staticmethod
    def get_lmks_acc(dlib_lmks: dict, isf_lmks: dict) -> float:
        bias = 0
        for key in isf_lmks.keys():
            if key not in dlib_lmks.keys():
                continue

            standard = isf_lmks[key]
            # print('標準: {}'.format(standard))
            xs = standard[0]
            ys = standard[1]

            ans = dlib_lmks[key]
            # print('DL: {}'.format(ans))
            x = ans[0]
            y = ans[1]

            # x
            x_bias = abs(xs - x) / xs
            y_bias = abs(ys - y) / ys
            # print(x_bias, y_bias)
            # print()

            bias += x_bias + y_bias

        return bias / 6
