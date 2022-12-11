# Success
import cv2, dlib, sys
import numpy as np

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    
    if(ret):
        dst = cv2.flip(frame, 1)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1,4)

        for(x,y,w,h) in faces:
           cv2.rectangle(dst, (x,y), (x+w, y+h), (0,0,255), 2)

        cv2.imshow('bilateral symmetry', dst)
        if cv2.waitKey(1) == ord('q'):
            break

    
cap.release()
cv2.destroyAllWindows()

        