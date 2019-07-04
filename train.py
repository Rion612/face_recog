import cv2
import numpy as np


face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recog = cv2.face.LBPHFaceRecognizer_create()

cap = cv2.VideoCapture(0)

y_labels = []
x_trains = []

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y) , (x+w,y+h), (255,0,0) , 2 )

        #cv2.imwrite("person%d.jpg" % cnt, gray[y-20:y+h+20, x-20:x+w+20] )

        x_trains.append( np.array(gray,"uint8") [y-20:y+h+20, x-20:x+w+20] )
        y_labels.append(1)

        print(np.array(gray,"uint8") [ y -20 : y + (h+20) , x -20 : x + (w+20) ])


    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



recog.train(x_trains,np.array(y_labels))
recog.save("xack.yml")