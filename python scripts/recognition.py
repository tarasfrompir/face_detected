#!/usr/bin/env python
# -*- coding: utf-8 -*-

# импортируем для получения времени
import time
import os
# Import OpenCV2 for image processing
import cv2
# импортируем для получения аргументов
import sys
# Для отправки данніх
import requests
import urllib

path = "/_majordomo/apps/ocv"
os.chdir(path)

#proverka na fail
if not os.path.exists('users.dat'):
    print ("netu vvedennih polyzovateley")
    exit

#работа с аргументами
# 1 argument nomer porta
portout=sys.argv[1] # так как нулевой аргумент это названия файла
# 2 argument nazvanie komnaty
rumname=sys.argv[2]
# 3 argument porog
porog=sys.argv[3] 
# 4 argument vremya raspoznavaniya
timetochek=sys.argv[4]
# 5 argument sootnoshenie kadra
waitingsize=sys.argv[5]
# 6 argument potok camery
camnumber=sys.argv[6]

# октрываем файл для определения пользователей
users = open('users.dat', 'r')
# считиваем пользователей в список
usersname = users.read().split("\n")[:-1]
users.close()

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=5, grid_x=16, grid_y=16)
# Using prebuilt frontal face training model, for face detection
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# Load the trained mode
recognizer.read('trainer/trainer.yml')
# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"
# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);
# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# устанавливаем стартовое время для таймера проверки лица
starttime = time.time()

# если длинна слова равно 1 символу то превращаем номер камеры в число
if len(camnumber)==1:
    camnumber = int(camnumber)
else:
    camnumber = str(camnumber)
	
# Start capturing video 
cam = cv2.VideoCapture(camnumber)
kadr = time.time()
while time.time() - starttime <float(timetochek):
    # Read the video waiting
    ret,image = cam.read()
    if ret == True and time.time() - kadr > 0.25:
        # обрабатіваем картинку ресайзим ее
        height, width = image.shape[:2]       
        width=1024
        if waitingsize == '16:9':
            height = int(width / 16 * 9)
        else:
            height = int(width / 4 * 3)  
        image = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
        cv2.waitKey(10)
		# Convert waiting to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get all face from the video waiting
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80), flags = 0)
        stop = 0
        UserNumber = 0
        # For each face in faces
        for(x,y,w,h) in faces:
            UserNumber += 1
            # Create rectangle around the face
            cv2.rectangle(image, (x-10,y-10), (x+w+10,y+h+10), (0,255,0), 4)
            # Recognize the face belongs to which ID
            prediction, conf= recognizer.predict(gray[y:y+h,x:x+w])
            face_id = usersname[int(prediction)]
            face_id = str(face_id[0:-3])
            print (face_id, conf)
            if int(conf)< int(porog): 
                rna = requests.get('http://127.0.0.1:'+str(portout)+'/objects/?object='+str(rumname)+'&op=set&p=User'+str(UserNumber)+'&v='+(str(face_id)))
                stop = 1
        if stop == 1:
            break
        # Display the video waiting with the bounded rectangle
        cv2.imshow('image',image) 
        kadr = time.time()
 
# Full stopprocess cam read
# Stop the camera
cam.release()
# Close all windows
cv2.destroyAllWindows()

