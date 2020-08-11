# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:14:12 2020

@author: HP
"""

import cv2
import numpy as np

import os

cap = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier(r"C:\Users\HP\Desktop\Face recognisation\haarcascade_frontalface_alt.xml")

name = input("Enter your name : ") #proving the name
#After that you have to Grab the image

frames = [] #input list
outputs = [] #output list 

while True:

    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame) 

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        cv2.imshow("My Screen", frame)
        cv2.imshow("My Face", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    if key == ord("c"): # when we press c it will create the capture and save the file with your name
        # cv2.imwrite(name + ".jpg", frame) #It is file name
        frames.append(gray.flatten()) #but we do not want to save,but we have to push (input & output System)
        #Append the flat version of gray
        outputs.append([name]) #in output I have to append my name
        
        #We have to convert frames(input) and output and concatenate them

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X]) # we are stacking them horizontally
#print(data.shape)

f_name ="face_data.npy"

if os.path.exists(f_name): # checking that file exists or not
    old = np.load(f_name) # load the old data
    data = np.vstack([old, data]) #add new data in bottom of old data i.e., vertical stacking(Complete data)

np.save(f_name, data)  

cap.release()
cv2.destroyAllWindows()