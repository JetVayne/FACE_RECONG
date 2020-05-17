# coding: utf-8
import cv2
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import numpy

from jf_ult.display_tool import DisplayTool


class PupilHelper:
    def __init__(self, image: numpy.ndarray):
        self.image = image
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = '../JayChou/models/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)

        # record the process result
        self.fetch_result = None

        # setting
        self.eye_threshold = 0.2  # 若資料集的人眼睛都偏小，可以再往下調
        self.x_padding = 5
        self.y_padding = 3

        # img var
        self.blurred = cv2.GaussianBlur(self.image, (11, 11), 0)  # 高斯模糊
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        ### eyes data ###
        self.EAR_mean = 0
        # left eye rectangle (ler)
        self.ler_x = None
        self.ler_y = None
        self.ler_w = None
        self.ler_h = None
        # right eye rectangle (rer)
        self.rer_x = None
        self.rer_y = None
        self.rer_w = None
        self.rer_h = None
        # left eye part
        self.left_ep_gray = None
        self.left_ep_color = None
        # right eye part
        self.right_ep_gray = None
        self.right_ep_color = None

        # pupil coord - 2D
        self.left_pupil_coord = None
        self.right_pupil_coord = None

    @staticmethod
    def eye_aspect_ratio(eye):
        """compute the euclidean distances between the two sets of
        vertical eye landmarks (x, y)-coordinates"""
        dist_15 = dist.euclidean(eye[1], eye[5])
        dist_24 = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        dist03 = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (dist_15 + dist_24) / (2.0 * dist03)

        # return the eye aspect ratio
        return ear

    def get_face_block(self):
        return self.detector(self.gray, 0)

    def load_eyes(self, face_block: dlib.rectangles):
        face = face_block[0]
        facial_landmarks = face_utils.shape_to_np(self.predictor(self.gray, face))

        l_eye_sti, l_eye_edi = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        r_eye_sti, r_eye_edi = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        left_eye = facial_landmarks[l_eye_sti:l_eye_edi]
        right_eye = facial_landmarks[r_eye_sti:r_eye_edi]

        # Eye Aspect Ratio (EAR)
        left_EAR = PupilHelper.eye_aspect_ratio(left_eye)
        right_EAR = PupilHelper.eye_aspect_ratio(right_eye)
        self.EAR_mean = (left_EAR + right_EAR) / 2

        ## Location of left bounding box
        self.ler_x, self.ler_y = left_eye[0][0], left_eye[2][1]  # left eye rectangle (ler)
        self.ler_w, self.ler_h = left_eye[3][0], left_eye[4][1]

        ## Location of Right bounding box
        self.rer_x, self.rer_y = right_eye[0][0], right_eye[2][1]  # right eye rectangle (rer)
        self.rer_w, self.rer_h = right_eye[3][0], right_eye[4][1]

        ## Extracting region of left eye for further process
        ler_y_region_st = self.ler_y + self.y_padding  # yLeft + PADDING_Y
        ler_y_region_ed = self.ler_h  # heightL
        ler_x_region_st = self.ler_x + self.x_padding  # xLeft + PADDING_X
        ler_x_region_ed = self.ler_w - self.x_padding  # widthL - PADDING_X
        # left eye part gray
        self.left_ep_gray = self.gray[ler_y_region_st: ler_y_region_ed, ler_x_region_st: ler_x_region_ed]
        # left eye part color
        self.left_ep_color = self.image[ler_y_region_st: ler_y_region_ed, ler_x_region_st: ler_x_region_ed]

        ## Extracting region of right eye for further process
        rer_y_region_st = self.rer_y + self.y_padding  # yRight + PADDING_Y
        rer_y_region_ed = self.rer_h - self.y_padding  # heightR - PADDING_Y
        rer_x_region_st = self.ler_x + self.x_padding  # xRight + PADDING_X
        rer_x_region_ed = self.ler_w - self.x_padding  # widthR - PADDING_X
        # right eye part gray
        self.right_ep_gray = self.gray[rer_y_region_st: rer_y_region_ed, rer_x_region_st: rer_x_region_ed]
        # right eye part color
        self.right_ep_color = self.image[rer_y_region_st: rer_y_region_ed, rer_x_region_st: rer_x_region_ed]

        # Verify that eyes are not closed
        if self.EAR_mean >= self.eye_threshold:
            # finding location of darker pixel inside eye region
            _, _, self.left_pupil_coord, _ = cv2.minMaxLoc(self.left_ep_gray)
            _, _, self.right_pupil_coord, _ = cv2.minMaxLoc(self.right_ep_gray)
        else:
            print('[WARN] the eye is smaller than threshold ! EAR-Mean: {}, Threshold: {}'.format(self.EAR_mean,
                                                                                                  self.eye_threshold))

    def show_pupil(self):
        face_block = self.get_face_block()

        if len(face_block) != 1:
            self.fetch_result = False
            return

        self.load_eyes(face_block)

        # !!! Q: 取到的座標是 部分眼睛區域的座標， 我想知道這個位置對於整張圖是哪裡 ?
        cv2.circle(self.left_ep_color, self.left_pupil_coord, 1, (0, 0, 255), 2)
        cv2.imshow('', self.image)
        cv2.waitKey(0)


ii = cv2.imread('man1.jpg')
x = PupilHelper(ii)
x.show_pupil()
