# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:09:44 2022
Detector de velocidad en vehiculos
@author: jorgil
"""

from itertools import count
from sqlite3 import register_converter
import cv2
import numpy as np
import time
from time import sleep

cap=cv2.VideoCapture('C:/Users/Python Scripts/ContadorCarros/SpeedDetector/carros.mp4')

count_line_position=530
count_line_position2=450

min_width_react=60
min_hight_react=60

subst=cv2.createBackgroundSubtractorMOG2(history=100)

def centro_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[]
offset=5  #error permitible entre pixeles
counter=0
counter2=0
fin=0
star=[]
fin=[]
calculo=0
vel=0
car_counter=0
pos1=0
pos2=0

while True:
    ret,original=cap.read()
    
    tempo=float(1/30)
    sleep(tempo)

    area_pts = np.array([[0,original.shape[0]-50],[540, 300], [original.shape[1]-650,300], [original.shape[1]-780, original.shape[0]-1]])
    
    imAux = np.zeros(shape=(original.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(original, original, mask=imAux)

    frame=image_area
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    dilat = subst.apply(blur)
    dilat=cv2.dilate(dilat,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fgmask = cv2.morphologyEx(dilat, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    dilatada=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    cv2.imshow('Filtro Morfologico 1',
               dilatada)
    contador,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(original,(310,count_line_position2),(575,count_line_position2),(255,127,127),3)
    cv2.line(original,(195,count_line_position),(560,count_line_position),(255,127,127),3)
    for (i,c) in enumerate(contador):

        (x,y,w,h)=cv2.boundingRect(c)
        
        if cv2.contourArea(c) > 3000:
            print("area de contorno: ",cv2.contourArea(c))
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 1)
            if count_line_position2-offset < y < count_line_position2+offset:
                car_counter = car_counter + 1
                pos1=x
                pos2=y-10           
        valcounter=(w>=min_width_react)and (h>=min_hight_react)
        if not valcounter:
            continue
        cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
        center = centro_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(original,center,4,(0,0,255),-1)
        #detección uno
        for (x,y) in detect:
            
            if y<(count_line_position2+offset) and y>(count_line_position2-offset):
                star.append(time.time())
                counter+=1
                cv2.line(original,(25,count_line_position2),(1200,count_line_position2),(0,127,255),3)
                detect.remove((x,y))
        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter2+=1
                fin.append(time.time())
                cv2.line(original,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                detect.remove((x,y))
                valor=fin[calculo]-star[calculo]
                calculo+=1
                vel=(10/valor)*(3.6)
        
        cv2.putText(original,'Speed:'+str(int(vel)),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)

    cv2.putText(original,"Total:"+str(counter),(450,70),cv2.FONT_HERSHEY_TRIPLEX,1.5,(55,40,250),3)
    cv2.putText(original,"Last Speed:"+str(int(vel)),(450,120),cv2.FONT_HERSHEY_TRIPLEX,1.5,(255,140,50),3)
    cv2.line(original, (295, 150), (315, 150), (0, 255, 0), 3)         # Visualización del conteo de autos
    cv2.drawContours(original, [area_pts], -1, (0, 0, 255), 2)
    cv2.rectangle(original, (original.shape[1]-70, 215), (original.shape[1]-5, 270), (0, 255, 0), 2)
    cv2.putText(original, str(counter), (original.shape[1]-55, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow('Freeway Speed',original)



    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()