import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
  
#image = cv2.imread('images4.jpg')

capture=cv2.VideoCapture('http://192.168.29.63:8080/video')

while True:
    _,frame=capture.read()

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
    faces = face_cascade.detectMultiScale(grayImage)
        
    if len(faces) == 0:
        print("No faces found")
      
    else:
        print("faces")
        print("faces.shape")
        print("Number of faces detected: " + str(faces.shape[0]))      
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
      
        cv2.rectangle(frame, ((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)
        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
      
cv2.imshow('Image with faces',frame)
capture.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


