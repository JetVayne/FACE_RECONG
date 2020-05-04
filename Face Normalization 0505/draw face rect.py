import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner

face_path = '295982.jpg'
predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=256)

image = cv2.imread(face_path)
image = imutils.resize(image, width=800)  # 改變圖片大小
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

color_list = [(0, 0, 255), (0, 255, 0)]

rects = detector(gray, 2)

rect_count = 0
for rect, color_code in zip(rects, color_list):
    rect_count += 1
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    text = 'FRect-{}'.format(rect_count)
    cv2.putText(image, text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_code, 1, cv2.LINE_AA)
    cv2.rectangle(image, (x, y), (x + w, y + h), color_code, 2)

# cv2.imshow('', image)
# cv2.waitKey(0)
output_name = '{}-rect-display.jpg'.format(face_path.split('.')[0])
cv2.imwrite(output_name, image)
