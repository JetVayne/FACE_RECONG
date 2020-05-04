import cv2
import imutils
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from imutils import face_utils

predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
face_path = '292455.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread(face_path)
image = imutils.resize(image, width=800)  # 改變圖片大小
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Input', image)
rects = detector(gray, 2)

print(len(rects))

for rect in rects:
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    faceAligned = fa.align(image, gray, rect)

    norm_img = np.zeros((800, 800))
    norm_img = cv2.normalize(faceAligned, norm_img, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Aligned", faceAligned)
    cv2.imshow('Norm', norm_img)
    cv2.imwrite('output.jpg', norm_img)
