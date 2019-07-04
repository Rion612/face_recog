import cv2

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("xack.yml")

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y) , (x+w,y+h), (255,0,0) , 2 )
        id ,conf = recog.predict(gray[y-20:y+h+20, x-20:x+w+20])

        if conf >= 50 and conf <=100 :
            name = str()
            if id == 1 : name = "xack"
            cv2.putText(frame,name,(x-30,y-30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1,cv2.LINE_AA)

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

