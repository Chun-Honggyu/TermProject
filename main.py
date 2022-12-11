import cv2, dlib, sys
import cv2 as cv
import numpy as np

# face detector
detector = dlib.get_frontal_face_detector()
# 68 points predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# distinguish points in each area of the face
all = list(range(0, 68))
jawline = list(range(0, 17))
right_eyebrow = list(range(17, 22))
left_eyebrow = list(range(22, 27))
nose = list(range(27, 36))
right_eye = list(range(36, 42))
left_eye = list(range(42, 48))
mouth_outline = list(range(48, 61))
mouth_inline = list(range(61, 68))

index = all

while True:
    
    # read the frame
    ret, frame = cap.read()
    
    if(ret):
        dst = cv2.flip(frame, 1)
        # convert to RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the frame
        detect = detector(gray, 1)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for face in detect:
            
            # find 68 points on the face
            shape = predictor(frame, face)

            lists = []
            for p in shape.parts():
                lists.append([p.x, p.y])

            lists = np.array(lists)

            # compute center and boundaries of face
            top_left = np.min(lists, axis=0)
            bottom_right = np.max(lists, axis=0)

            center_x, center_y = np.mean(lists, axis=0).astype(np.int)

            for s in lists:
                cv2.circle(frame, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(frame, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(frame, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(frame, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            for i, point in enumerate(lists[index]):
                points = (point[0], point[1])
                cv.circle(frame, points, 2, (0,255,0), -1)

            for(x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # show the frame
        cv2.imshow('result', frame)

        # if the 'q' key was pressed, break from the loop
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()