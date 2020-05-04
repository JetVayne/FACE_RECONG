# coding: utf-8
import cv2
import imutils
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from imutils import face_utils
import os


class DlibNrlz:
    def __init__(self, img_dir_path: str, model_code: int):
        self.dir = img_dir_path
        self.resize_width = 800

        if model_code == 1:
            self.predictor_path = 'models/shape_predictor_5_face_landmarks.dat'
        elif model_code == 2:
            self.predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

        self.img_list = list()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.face_aligner = FaceAligner(self.predictor, desiredFaceWidth=256)

    def launch(self):
        self.load_img_data()
        self.nrlz_helper()

    def load_img_data(self):
        self.img_list = [self.dir + '/' + fn for fn in os.listdir(self.dir) if os.path.isfile(self.dir + '/' + fn)]

    def nrlz_helper(self):
        for img_path in self.img_list:
            x = self.get_face_block(img_path)

            if len(x) != 1:
                print('臉部方框偵測異常', img_path, x)


    def get_face_block(self, img_path: str) -> dlib.rectangles:
        image = cv2.imread(img_path)
        image = imutils.resize(image, width=self.resize_width)  # 改變圖片大小
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 2)

        return rects


qqq = 'Data/A/img'
nnn = DlibNrlz(qqq, 2)
nnn.launch()
