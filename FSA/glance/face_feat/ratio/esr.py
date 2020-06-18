# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool


class ESR:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        # calculate
        self.left_eye_head = landmarks['43']
        self.left_eye_tail = landmarks['46']

        self.right_eye_head = landmarks['40']
        self.right_eye_tail = landmarks['37']

        self.left_eye_long = GeomTool.get_pt_dist(self.left_eye_tail, self.left_eye_head)
        self.right_eye_long = GeomTool.get_pt_dist(self.right_eye_tail, self.right_eye_head)

        self.eye_bar = GeomTool.get_pt_dist(self.left_eye_tail, self.right_eye_tail)
        self.eye_dist = GeomTool.get_pt_dist(self.left_eye_head, self.right_eye_head)

        self.val = (self.eye_bar - self.eye_dist) / self.eye_dist
        self.val_2 = (self.left_eye_long + self.right_eye_long) / self.eye_dist
        self.val_3 = (self.left_eye_long + self.right_eye_long) / (self.eye_dist * 2)
        self.val_4 = self.eye_bar / self.eye_dist

        # display
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.thick = 2

    def get_face_coords(self):
        coords = []
        for i in range(17):
            lm_id = str(i + 1)
            coords.append(self.landmarks[lm_id])

        return coords

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.left_eye_tail, self.right_eye_tail, self.red, self.thick)
        cv2.line(temp, self.left_eye_head, self.right_eye_head, self.green, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
