import cv2 as cv

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grey, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv.imshow('Face Detection', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()