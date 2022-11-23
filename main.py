import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 255)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    cv2.imshow("Video", cv2.flip(laplacian, 1))

capture.release()
cv2.destroyAllWindows()
