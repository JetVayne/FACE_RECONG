# coding: utf-8
import numpy
import cv2
from imutils import face_utils
import sys


class DisplayTool:
    # line thickness
    line_thick = 1

    # dot_size
    dot_size = 3

    # solid
    solid = -1

    # color code (cc)
    red_cc = (0, 0, 255)
    green_cc = (0, 255, 0)
    blue_cc = (255, 0, 0)
    pink_cc = (255, 223, 220)

    @staticmethod
    def get_color(color_str: str):
        if color_str == 'red':
            return DisplayTool.red_cc
        elif color_str == 'blue':
            return DisplayTool.blue_cc
        elif color_str == 'green':
            return DisplayTool.green_cc
        elif color_str == 'pink':
            return DisplayTool.pink_cc
        else:
            return DisplayTool.red_cc

    @staticmethod
    def show_landmarks(img: numpy.ndarray, landmarks: dict, dest_path=None, color=red_cc, size=dot_size, style=solid):
        # if you want to output an img instead of display it, just add the dest path
        window_name = sys._getframe().f_code.co_name
        temp = img.copy()
        if isinstance(color, str):
            color = DisplayTool.get_color(color)

        for lm in landmarks:
            pos = landmarks[lm]
            cv2.circle(temp, pos, size, color, style)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow(window_name, temp)
            cv2.waitKey(0)

    @staticmethod
    def show_rect(img: numpy.ndarray, rect=None, mode='dlib', coord=None, color=red_cc, thick=line_thick):
        window_name = sys._getframe().f_code.co_name
        temp = img.copy()

        # change color
        if isinstance(color, str):
            color = DisplayTool.get_color(color)

        if mode == 'dlib':
            x, y, w, h = face_utils.rect_to_bb(rect)
            cv2.rectangle(temp, (x, y), (x + w, y + h), color, thick)
            cv2.imshow(window_name, temp)
            cv2.waitKey(0)
        else:
            x = coord[0]
            y = coord[1]
            w = coord[2]
            h = coord[3]
            if mode == 'cod1':
                cv2.rectangle(temp, (x, y), (x + w, y + h), color, thick)
                cv2.imshow(window_name, temp)
                cv2.waitKey(0)

            elif mode == 'cod2':
                cv2.rectangle(temp, (x, y), (w, h), color, thick)
                cv2.imshow(window_name, temp)
                cv2.waitKey(0)

    @staticmethod
    def show_point(img: numpy.ndarray, coord: tuple, color=red_cc, size=dot_size, style=solid):
        window_name = sys._getframe().f_code.co_name
        # just for single point
        temp = img.copy()
        # change color
        if isinstance(color, str):
            color = DisplayTool.get_color(color)

        cv2.circle(temp, coord, size, color, style)
        cv2.imshow(window_name, temp)
        cv2.waitKey(0)

    @staticmethod
    def show_pupils(img: numpy.ndarray, left_pc: tuple, right_pc: tuple, color=red_cc, size=dot_size, style=solid):
        window_name = sys._getframe().f_code.co_name
        temp = img.copy()
        # change color
        if isinstance(color, str):
            color = DisplayTool.get_color(color)

        cv2.circle(temp, left_pc, size, color, style)
        cv2.circle(temp, right_pc, size, color, style)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
