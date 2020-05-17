# coding: utf-8
import dlib
import cv2
import numpy
import os


class FBHelper:
    cascade_path = '../models/haarcascade_frontalface_default.xml'
    cas_classifier = cv2.CascadeClassifier(cascade_path)
    cas_classifier.load('E:/PycharmProjects/ImgNrlz/module/models/haarcascade_frontalface_default.xml')
    dlib_detector = dlib.get_frontal_face_detector()

    @staticmethod
    def get_face_block(img: numpy.ndarray) -> dlib.rectangle:
        # Priority - 1: dlib
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_block = FBHelper.dlib_detector(gray, 2)
        if len(face_block) == 1:
            return face_block[0]

        # Priority - 2: Haar cascade
        face_block = FBHelper.cas_classifier.detectMultiScale(img, 1.3, 8)
        if len(face_block) == 1:
            x, y, w, h = face_block[0]
            return dlib.rectangle(x, y, x + w, y + h)
