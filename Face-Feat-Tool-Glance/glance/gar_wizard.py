# coding: utf-8
from deepface import DeepFace
import numpy
import cvlib
from glance.jf_ult.log_tool import LogTool
from glance.jf_ult.fb_helper import FBHelper
import cv2


class GarWizard:
    @staticmethod
    def get_face_dict(img_path: str) -> dict:
        try:
            return DeepFace.analyze(img_path)
        except Exception as e:
            msg = LogTool.pp_exception(e)
            print(msg)

    @staticmethod
    def get_gender(image: numpy.ndarray):
        padding = 20
        try:
            face_block = FBHelper.get_face_block(image)

            if face_block:
                startX, startY = max(0, face_block.left() - padding), max(0, face_block.top() - padding)

                endX, endY = min(image.shape[1] - 1, face_block.right() + padding), min(image.shape[0] - 1,
                                                                                        face_block.bottom() + padding)

                face_grid = numpy.copy(image[startY:endY, startX:endX])

                label, confidence = cvlib.detect_gender(face_grid)
                idx = numpy.argmax(confidence)

                if label[idx] == 'male':
                    return 'Male'
                else:
                    return 'Female'

        except Exception as e:
            print('Failed to get gender from CVLib.  ERROR: {}'.format(e))
