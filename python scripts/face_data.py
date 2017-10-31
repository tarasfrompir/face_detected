#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import configparser
import os
import datetime
# импортируем для получения аргументов
import sys
# Import numpy for matrix calculation
import numpy as np
# Import Python Image Library (PIL)
from PIL import Image
# Import OpenCV2 for image processing
import cv2
# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


path = "/_majordomo/apps/ocv"

#работа с аргументами
# 2 argument адрес камеры
camnumber=sys.argv[2] # так как нулевой аргумент это названия файла
# 1 argument имя пользователя
face_iduser=sys.argv[1]


# если длинна слова равно 1 символу то превращаем номер камеры в число
if len(camnumber)==1:
    camnumber = int(camnumber)
else:
    camnumber = str(camnumber)

cam = cv2.VideoCapture(camnumber)


# Detect object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#если файла нет то создаем его
if not os.path.exists('users.dat'):
    users = open('users.dat', 'w')
    users.write('\n')
    users.close()
#если есть ищем имя пользователя
face_id = 0
ext = 0    
users = open('users.dat', 'r')
for line in users.readlines():
    if line.find(face_iduser)>-1:
        face_id = int(line[-2])
        ext = 1
        break
    else:
        face_id += 1
users.close()

# иначе записываем его в файл
if ext == 0:
    users = open('users.dat', 'at')
    users.write(str(face_iduser) + "  " + (str(face_id))+'\n')
    users.close()


# Initialize sample face image
count = 0

# Start looping
while(True):

    # Capture video frame
    ret, im = cam.read(5)
    
    if ret== True :
        #отключено в связи с голым спользованием вебки для внесения данных
        # обрабатіваем картинку ресайзим ее
        #height, width = im.shape[:2]
        #if int(width) > 1024:
        #   width = 1024
        #if framesize == '16:9':
        #    height = width / 16 * 9
        #else:
        #    height = width / 4 * 3  
        #im = cv2.resize(im, (int(width), int(height))) 

        # Convert frame to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray,
                                               scaleFactor=1.05,
                                               minNeighbors=8,
                                               minSize=(150, 150),
                                               flags = 0)

        # Loops for each faces
        for (x,y,w,h) in faces:

            # Crop the image frame into rectangle
            cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
            
            # Increment sample face image
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    # Display the video frame, with bounded rectangle on the person's face
    cv2.imshow('frame', im)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>99:
        break

# Stop video
cam.release()

# Close all started windows
cv2.destroyAllWindows()
