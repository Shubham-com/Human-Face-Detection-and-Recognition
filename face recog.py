# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:55:44 2020

@author: HP
"""

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy")

print(data.shape, data.dtype)

X = data[:, 1:].astype(int)
y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Face recognisation\haarcascade_frontalface_alt.xml")
while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()]) #put gray scale item in flatten 

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # we have to create rectangle

            cv2.putText(frame, str(out[0]), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2) # put neme in the frame

            print(out)

            cv2.imshow("My Face", gray)

            cv2.imshow("My Screen", frame)


    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()