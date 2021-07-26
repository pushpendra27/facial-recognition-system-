import numpy as np
import numpy
import imutils
import cv2
import subprocess as sp
import os
from imutils.object_detection import non_max_suppression

videoFile = "/home/pushpendra/out"
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))


#Using Ip camera or Rtsp camera for live Counting
cap = cv2.VideoCapture('http://192.168.29.63:8080/video')

index = 0
frameRate = cap.get(cv2.CAP_PROP_FPS)
print(frameRate)
import time
start = time.time()
import csv
with open('people.csv', 'w', ) as csvfile:
    peoplewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    while cap.isOpened():
        curr_frame = cap.get(1)
        print("frame: ", curr_frame)

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray )

        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4),
                padding=(8, 8), scale=1.05)
        
        # for (x, y, w, h) in rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # Write to csv, first column: frame number, 2nd column: no. of peoples
        peoplewriter.writerow([cap.get(1), len(pick)])
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.putText(frame, "Number of faces detected: " + str(weights.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 3,  (0,0,0), 1)
            res=cv2.resize(frame,(960,540))
            cv2.imshow('orig', res)
 
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
