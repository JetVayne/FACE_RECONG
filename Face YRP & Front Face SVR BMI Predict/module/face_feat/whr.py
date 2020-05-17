# coding: utf-8
import cv2
from module.jf_ult.geom_tool import GeomTool


class WHR:
    def __init__(self, landmarks: dict, img=None):
        self.landmarks = landmarks
        self.image = img

        ## calculate
        self.right_jaw = self.landmarks['5']
        self.left_jaw = self.landmarks['13']
        self.jaw_width = GeomTool.get_pt_dist(self.right_jaw, self.left_jaw)

        self.nasal_root = self.landmarks['28']
        self.prj_point = GeomTool.get_prj_point(self.nasal_root, (self.left_jaw, self.right_jaw))
        self.lower_face_height = GeomTool.get_pt_dist(self.nasal_root, self.prj_point)

        self.val = self.jaw_width / self.lower_face_height

        # display
        self.color = (0, 0, 255)
        self.thick = 2

    def show(self, dest_path=None):
        temp = self.image.copy()
        cv2.line(temp, self.nasal_root, self.prj_point, self.color, self.thick)
        cv2.line(temp, self.right_jaw, self.left_jaw, self.color, self.thick)

        if dest_path:
            cv2.imwrite(dest_path, temp)
        else:
            cv2.imshow('{}'.format(__name__), temp)
            cv2.waitKey(0)
