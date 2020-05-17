# coding: utf-8

import cv2
import math
import numpy
import numpy as np
import dlib
from module.jf_ult.fb_helper import FBHelper
import os


class FaceYRP:
    def __init__(self, image: numpy.ndarray, image_name='', mode='normal', dest_path=None):
        self.image = image
        self.image_name = image_name
        self.mode = mode
        self.dest_path = dest_path

        self.result = None

        # landmark module
        # self.predictor_path = 'models/shape_predictor_68_face_landmarks.dat'
        self.predictor_path = 'module/models/shape_predictor_68_face_landmarks.dat'
        x = os.path.exists(self.predictor_path)

        self.predictor = dlib.shape_predictor(self.predictor_path)

        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4, 1))

        # img project points, obj project points, nose_coord
        self.image_points = None
        self.img_prj_points = None
        self.obj_prj_points = None
        self.nose_coord = None

        # analysis result
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    @staticmethod
    def show_face_block(image: numpy.ndarray, face_block: dlib.rectangle):
        img_temp = image.copy()
        cv2.rectangle(img_temp, (face_block.left(), face_block.top()),
                      (face_block.right(), face_block.bottom()), (255, 0, 255), 2)
        cv2.imshow('', img_temp)
        cv2.waitKey(0)

    @staticmethod
    def get_image_points(landmarks: dict) -> numpy.ndarray:
        return np.array([
            landmarks[31],  # Nose tip
            landmarks[9],  # Chin
            landmarks[46],  # Left eye left corner
            landmarks[37],  # Right eye right corner
            landmarks[55],  # Left Mouth corner
            landmarks[49]  # Right mouth corner
        ], dtype="double")

    @staticmethod
    def get_object_points() -> numpy.ndarray:
        return np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-165.0, 170.0, -135.0),  # Left eye left corner
            (165.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

    @staticmethod
    def get_camera_matrix(img: numpy.ndarray) -> numpy.ndarray:
        img_size = img.shape  # (height, width, color_count) (306, 620, 3)
        center = (img_size[1] / 2, img_size[0] / 2)  # (x_mid, y_mid)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        return camera_matrix

    def get_landmarks(self, img: numpy.ndarray, face_block: dlib.rectangle) -> dict:
        result = dict()
        face_points = numpy.matrix([[p.x, p.y] for p in self.predictor(img, face_block).parts()])
        for index, point in enumerate(face_points):
            pos = (point[0, 0], point[0, 1])
            result[index + 1] = pos

        return result

    def get_orientation(self, img: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray, tuple, tuple):
        face_block = FBHelper.get_face_block(img)

        if face_block is None:
            print('[WARN] - Failed to fetch face block, img: {}'.format(self.image_name))
            self.result = False
            return

        # FaceYRP.show_face_block(self.image, face_block)

        landmarks = self.get_landmarks(img, face_block)

        self.image_points = FaceYRP.get_image_points(landmarks)
        object_points = FaceYRP.get_object_points()
        camera_matrix = FaceYRP.get_camera_matrix(img)

        success, rotation_vector, translation_vector = cv2.solvePnP(object_points, self.image_points,
                                                                    camera_matrix, self.dist_coeffs,
                                                                    flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])

        self.img_prj_points, img_jac = cv2.projectPoints(axis, rotation_vector, translation_vector,
                                                         camera_matrix, self.dist_coeffs)

        self.obj_prj_points, obj_jac = cv2.projectPoints(object_points, rotation_vector, translation_vector,
                                                         camera_matrix, self.dist_coeffs)

        rtv_matrix = cv2.Rodrigues(rotation_vector)[0]
        # print(rtv_matrix, translation_vector)
        prj_matrix = np.hstack((rtv_matrix, translation_vector))

        euler_angles = cv2.decomposeProjectionMatrix(prj_matrix)[6]  # 歐拉角

        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
        self.pitch = math.degrees(math.asin(math.sin(pitch)))  # 上下
        self.roll = - math.degrees(math.asin(math.sin(roll)))  # 繞園轉
        self.yaw = math.degrees(math.asin(math.sin(yaw)))  # 左右

        self.nose_coord = landmarks[31]

        # print(self.pitch, self.roll, self.yaw)
        self.result = True

        return self.img_prj_points, self.img_prj_points, \
               (int(self.roll), int(self.pitch), int(self.yaw)), \
               self.nose_coord, self.image_points

    def launch(self):
        self.get_orientation(self.image)

        if self.mode != 'normal':
            self.draw_yrp_line()

    def draw_yrp_line(self):
        cv2.line(self.image, self.nose_coord, tuple(self.img_prj_points[1].ravel()), (0, 255, 0), 3)  # GREEN
        cv2.line(self.image, self.nose_coord, tuple(self.img_prj_points[0].ravel()), (255, 0, 0), 3)  # BLUE
        cv2.line(self.image, self.nose_coord, tuple(self.img_prj_points[2].ravel()), (0, 0, 255), 3)  # RED

        # lm_color = (208, 216, 129)  # Tiffany Blue
        # for lm in self.image_points:
        #     x, y = int(lm[0]), int(lm[1])
        #     cv2.circle(self.image, (x, y), 3, lm_color, -1)

        dimension = ['Roll', 'Pitch', 'Yaw']
        rotate_degree = [self.roll, self.pitch, self.yaw]

        for i in range(len(dimension)):
            dim = dimension[i]
            d = float(rotate_degree[i])
            d = round(d, 2)

            row_base = 30
            text_x = 10
            text_y = (row_base * i) + 30
            text_size = 0.5
            text_color = (161, 0, 161)

            cv2.putText(self.image, '{} : {}'.format(dim, d), (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        text_size, text_color, 1, cv2.LINE_AA)

        cv2.imwrite(self.dest_path, self.image)
