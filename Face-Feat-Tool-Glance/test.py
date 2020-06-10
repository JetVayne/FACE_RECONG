from pprint import pprint as pp
from glance.face_analyst import FaceAnalyst

img_path = 'puff.jpg'

fa = FaceAnalyst(img_path, gen_ff_img=True)
fa.analyze()

if fa.result:
    pp(fa.data)
