# coding: utf-8
from jf_ult.geom_tool import GeomTool


class FFTool:
    @staticmethod
    def get_CJWR(landmarks: dict):  # 1、17
        # get cheek width
        right_cheek = landmarks['1']
        left_cheek = landmarks['17']
        cheek_width = GeomTool.get_pt_dist(right_cheek, left_cheek)

        # get jaw width
        right_jaw = landmarks['4']
        left_jaw = landmarks['14']



