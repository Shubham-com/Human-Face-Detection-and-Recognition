# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 11:45:07 2020

@author: HP
"""

import cv2

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Face recognisation\haarcascade_frontalface_alt.xml")

while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w] # Here we are cutting the rectangle frame of our face
            fix = cv2.resize(cut, (100, 100)) # we are fixing the Scale of the face
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY) # Converting into black and white colour

            

            cv2.imshow("My Screen", frame)
            cv2.imshow("My Face", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()