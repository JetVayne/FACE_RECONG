# coding: utf-8
from deepface import DeepFace
from glance.jf_ult.log_tool import LogTool


class GarWizard:
    @staticmethod
    def get_face_dict(img_path: str) -> dict:
        try:
            return DeepFace.analyze(img_path)
        except Exception as e:
            msg = LogTool.pp_exception(e)
            print(msg)

