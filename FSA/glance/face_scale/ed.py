# coding: utf-8
import cv2
from glance.jf_ult.geom_tool import GeomTool


class ED:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        ## calculate
        self.left_eye_head = landmarks['43']
        self.right_eye_head = landmarks['40']
        self.eye_dist = GeomTool.get_pt_dist(self.left_eye_head, self.right_eye_head)

        self.nasal_root = landmarks['28']
        self.nose_tip = landmarks['31']
        self.nose_bridge = GeomTool.get_pt_dist(self.nasal_root, self.nose_tip)

        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.cheek_width = GeomTool.get_pt_dist(self.left_cheek, self.right_cheek)

        # vertical
        self.nose_bot = landmarks['34']
        self.moth_top = landmarks['52']
        self.philtrum = GeomTool.get_pt_dist(self.nose_bot, self.moth_top)

        # whole face vertical
        self.nasal_root = landmarks['28']
        self.jaw_tip = landmarks['9']
        self.face_height = GeomTool.get_pt_dist(self.nasal_root, self.jaw_tip)

        # self.val = (self.eye_dist * self.nose_bridge) / (self.cheek_width * self.face_height)
        self.val = self.eye_dist / self.cheek_width

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
        cv2.line(temp, self.left_cheek, self.right_cheek, self.red, self.thick)
        cv2.line(temp, self.right_eye_head, self.left_eye_head, self.green, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
