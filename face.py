import cv2
import dlib
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# distinguish points in each area of the face
jawline = list(range(0, 17))
left_eyebrow = list(range(22, 27))
right_eyebrow = list(range(17, 22))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
nose = list(range(27, 36))
mouth_outline = list(range(48, 61))
mouth_inline = list(range(61, 68))

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(100, 100), flags = cv2.CASCADE_SCALE_IMAGE)

    # find landmarks on face
    for(x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        landmarks_display = landmarks[0:68]

        # output points
        for idx, point in enumerate(landmarks_display):
            pos = point([0, 0], point(0, 1))
            cv2.circle(frame, pos, 2, color = (0, 255, 255), thickness = -1)

    return frame