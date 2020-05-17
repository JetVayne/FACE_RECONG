# coding: utf-8
import cv2
from module.jf_ult.geom_tool import GeomTool


class MEH:
    def __init__(self, landmarks: dict, img=None):
        # self.landmarks = landmarks
        self.image = img
        self.landmarks = landmarks

        ## calculate
        # right eye part
        self.right_eb_tail = self.landmarks['18']
        self.right_eye_tail = self.landmarks['37']
        self.right_ee_tail = GeomTool.get_pt_dist(self.right_eb_tail, self.right_eye_tail)

        self.right_eb_mid = self.landmarks['20']
        self.right_eyelid_mid = GeomTool.get_mid_point(self.landmarks['38'], self.landmarks['39'])
        self.right_ee_mid = GeomTool.get_pt_dist(self.right_eb_mid, self.right_eyelid_mid)

        self.right_eb_head = self.landmarks['22']
        self.right_eye_head = self.landmarks['40']
        self.right_ee_head = GeomTool.get_pt_dist(self.right_eb_head, self.right_eye_head)

        # left head part
        self.left_eb_tail = self.landmarks['27']
        self.left_eye_tail = self.landmarks['46']
        self.left_ee_tail = GeomTool.get_pt_dist(self.left_eb_tail, self.left_eye_tail)

        self.left_eb_mid = self.landmarks['25']
        self.left_eyelid_mid = GeomTool.get_mid_point(self.landmarks['44'], self.landmarks['45'])
        self.left_ee_mid = GeomTool.get_pt_dist(self.left_eb_mid, self.left_eyelid_mid)

        self.left_eb_head = self.landmarks['23']
        self.left_eye_head = self.landmarks['43']
        self.left_ee_head = GeomTool.get_pt_dist(self.left_eb_head, self.left_eye_head)

        self.val = (self.right_ee_tail + self.right_ee_mid + self.right_ee_head +
                    self.left_ee_tail + self.left_ee_mid + self.left_ee_head) / 6

        # display
        self.red = (0, 0, 255)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()

        cv2.line(temp, self.right_eb_tail, self.right_eye_tail, self.red, self.thick)
        cv2.line(temp, self.right_eb_mid, self.right_eyelid_mid, self.red, self.thick)
        cv2.line(temp, self.right_eb_head, self.right_eye_head, self.red, self.thick)

        cv2.line(temp, self.left_eb_tail, self.left_eye_tail, self.red, self.thick)
        cv2.line(temp, self.left_eb_mid, self.left_eyelid_mid, self.red, self.thick)
        cv2.line(temp, self.left_eb_head, self.left_eye_head, self.red, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
