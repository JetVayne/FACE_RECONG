# coding: utf-8
import cv2
from module.jf_ult.geom_tool import GeomTool


class CJWR:
    def __init__(self, landmarks: dict, img=None):
        self.landmarks = landmarks
        self.image = img

        ## calculate
        # cheek part
        self.left_cheek = self.landmarks['17']
        self.right_cheek = self.landmarks['1']
        self.cheek_width = GeomTool.get_pt_dist(self.left_cheek, self.right_cheek)

        # jaw part
        self.right_jaw = self.landmarks['5']
        self.left_jaw = self.landmarks['13']
        self.jaw_width = GeomTool.get_pt_dist(self.right_jaw, self.left_jaw)

        self.val = self.cheek_width / self.jaw_width

        # display
        self.color = (0, 0, 255)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.right_cheek, self.left_cheek, self.color, self.thick)
        cv2.line(temp, self.right_jaw, self.left_jaw, self.color, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
