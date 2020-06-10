# coding: utf-8
import pathlib

import cv2
import dlib
import numpy

from glance.jf_ult.log_tool import LogTool


class FBHelper:
    cascade_ps = '{}/models/haarcascade_frontalface_default.xml'.format(pathlib.Path(__file__).parent.parent)
    cascade_path = pathlib.Path(cascade_ps).as_posix()
    cas_classifier = cv2.CascadeClassifier()
    cas_classifier.load(cascade_path)

    dlib_detector = dlib.get_frontal_face_detector()

    @staticmethod
    def get_face_block(img: numpy.ndarray, mode='normal') -> dlib.rectangle:
        face_block = list()

        if mode == 'normal':
            # Priority - 1: dlib-colorful
            try:
                face_block = FBHelper.dlib_detector(img, 1)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                return face_block[0]

            # Priority - 2: dlib-gray
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_block = FBHelper.dlib_detector(gray, 2)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                return face_block[0]

            # Priority - 3: Haar cascade
            try:
                face_block = FBHelper.cas_classifier.detectMultiScale(img, 1.3, 8)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                x, y, w, h = face_block[0]
                return dlib.rectangle(x, y, x + w, y + h)

        elif mode == 'dlib':
            try:
                face_block = FBHelper.dlib_detector(img, 1)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                return face_block[0]

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                face_block = FBHelper.dlib_detector(gray, 2)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                return face_block[0]

        elif mode == 'haar':
            # Priority - 3: Haar cascade
            try:
                face_block = FBHelper.cas_classifier.detectMultiScale(img, 1.3, 8)
            except Exception as e:
                msg = LogTool.pp_exception(e)
                print(msg)

            if len(face_block) == 1:
                x, y, w, h = face_block[0]
                return dlib.rectangle(x, y, x + w, y + h)

    @staticmethod
    def get_width_height(face_block: dlib.rectangle) -> tuple:
        pass
