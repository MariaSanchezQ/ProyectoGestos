def calcula_gestos(puntos):
    gestos = dict() 
    #Calcula el promedio de los puntos de la muñeca
    prom_muneca = (puntos[:,0] + puntos[:,1] + puntos[:,2])/3
    #Calcula el promedio de los puntos de los nudillos
    prom_MCP = (puntos[:,5] + puntos[:,9] + puntos[:,13] + puntos[:,17])/4
    #Calcula el promedio de la punta de los dedos
    prom_TIP = (puntos[:,8] + puntos[:,12] + puntos[:,16] + puntos[:,20])/4

    #Calcula la distancia que existe entre la muñeca y la punta de los dedos
    dist_mun_TIP = np.linalg.norm(prom_muneca - prom_TIP)
    #Calcula la distancia que existe entre la muñeca y los nudillos
    dist_mun_MCP = np.linalg.norm(prom_muneca - prom_MCP)

    
    gestos['mano_abierta'] = dist_mun_TIP / (2 * dist_mun_MCP)
    gestos['mano_cerrada'] = 1 - dist_mun_TIP / (2 * dist_mun_MCP)
    return gestos

import time
import numpy as np 
import cv2
import mediapipe as mp 
from math import acos, degrees

c = ""
p = ""
mpHands = mp.solutions.hands
mp_dedos = mpHands.HandLandmark
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('arrojara1.mp4')  # Captura un 'stream' a partir de un archivo de video.
cap_s= int(cap.get(4)), int(cap.get(3)) # Obtiene las dimensiones del 'stream'.
print('Dimensiones del "stream"',cap_s)
pTime=time.time() 
varios_pun=[]

# el siguiente ciclo itera sobre los 'frames'
repetir=True
while repetir:
    gesto={} # se inicializa la variable que almacena gestos
    gesto['mano_abierta'] = 0
    gesto['mano_cerrada'] = 0
    
    success, img = cap.read()

    #Se realiza espejo para voltear el video
    #img =cv2.flip(img, 1) 
    height, width, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    results =hands.process(imgRGB) 

    if results.multi_hand_landmarks: 
        for handLms in results.multi_hand_landmarks:         

            x2 = int(handLms.landmark[0].x * width)
            y2 = int(handLms.landmark[0].y * height)

            x1 = int(handLms.landmark[9].x * width)
            y1 = int(handLms.landmark[9].y * height)

            x3 = int(x2 - 130)
            y3 = int(y2)

            p1 = np.array([x1,y1])
            p2 = np.array([x2,y2])
            p3 = np.array([x3,y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)


            #Angulo

            angle = degrees(acos((l1**2 + l3**2 - l2**2)/ (2*l1*l3)))

            aux_image = np.zeros(img.shape, np.uint8)
            cv2.line(aux_image, (x1,y1), (x2,y2), (255,255,0),20)
            cv2.line(aux_image, (x2,y2), (x3,y3), (255,255,0),20)
            cv2.line(aux_image, (x1,y1), (x3,y3), (255,255,0),5)
            contours = np.array([[x1,y1],[x2,y2],[x3,y3]])
            cv2.fillPoly(aux_image, pts=[contours],color=(120,0,250))

            output = cv2.addWeighted(img, 1, aux_image, 0.8,0)

            cv2.circle(output,(x1,y1),6,(215,48,12),4)
            cv2.circle(output,(x2,y2),6,(215,48,12),4)
            cv2.circle(output, (x3,y3),6,(215,48,12),4)
            cv2.putText(output, str(int(angle)),(x2 + 15,y2) , 1, 1.5, (255,255,255),2)
            cv2.imshow("aux_image",output)
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # dibuja la mano
            pos=[] # variable usada para almacenar los puntos de la mano
            for punto in handLms.landmark: # itera sobre cada punto (landmark)
                pos.append([punto.x,punto.y,punto.z]) # almacena  cada punto de la mano

            
            varios_pun.append(pos) 
            if len(varios_pun)>10: 
                varios_pun=varios_pun[-10:]
            varios_pun_np=np.median(np.array(varios_pun),axis=0).T

            gesto = calcula_gestos(varios_pun_np) 

            
            cv_pos_punto=(int(varios_pun_np[0,8]*cap_s[1]),int(varios_pun_np[1,8]*cap_s[0]))
            cv2.circle(img,cv_pos_punto, 5, (0,255,255), -1)
            
            
            if gesto['mano_abierta'] > 0.9:
                    cv2.circle(img,cv_pos_punto, 5, (0,0,255), -1)
                    c = "mano abierta"

            if gesto['mano_cerrada'] > 0.4:
                   cv2.circle(img,cv_pos_punto, 5, (0,255,0), -1)
                   c = "mano cerrada"


            if angle > 60:
                    p = "Impulso"

            if angle  > 14 and angle < 30:
                    p = "Comienzo"

            if angle < 30 and gesto['mano_abierta'] > 0.8 and y1 < y2 :
                    p = "Soltar"

            if angle >0 and angle < 50 and y1 > y2 and gesto['mano_abierta']>0.9:
                    p = "Arrojado"

            if angle > 30 and angle < 60 and gesto['mano_abierta'] > 0.8 and y1 < y2 :
                    p = "Arrojar enfrente"
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(p),(10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    #cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 2, (12,215,194),2)
    cv2.putText(img, str(c),(10,110),cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
    
    if cv2.waitKey(5) & 0xFF == 27: #esc
      break
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
