# Success
import cv2

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    
    if(ret):
        dst = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.imshow('bilateral symmetry', dst)
        if cv2.waitKey(1) == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()
