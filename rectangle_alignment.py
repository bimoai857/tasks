

import numpy as np
import matplotlib.pyplot as plt
import cv2

import math

img = cv2.imread('new_new.png')
immg=cv2.imread('new_new_new.png')
imgray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 100, 100, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours = " + str(len(contours)))



box1=[]

for i in range(2,12,3):
  box1.append(np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i]))))

box2=[]

for i in range(3,13,3):
  box2.append(np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i]))))


angles=[-30,30,-15,15]
for x,y in enumerate(angles):
  a=math.radians(y)
  x0=box1[x][3][0]
  y0=box1[x][3][1]

  transformed_points1=[]
  transformed_points2=[]

  for i in box1[x]:

    x_t = np.int0(((i[0] - x0) * math.cos(a)) - ((i[1] - y0) * math.sin(a)) + x0);
    y_t= np.int0(((i[0] - x0) * math.sin(a)) + ((i[1] - y0) * math.cos(a)) + y0);
    transformed_points1.append((x_t,y_t))

  orient=[(1,2),(0,1),(0,3),(0,1)]
  
  for i,j in enumerate(box2[x]):
      x_t = np.int0(((j[0] - x0) * math.cos(a)) - ((j[1] - y0) * math.sin(a)) + x0);
      y_t= np.int0(((j[0] - x0) * math.sin(a)) + ((j[1] - y0) * math.cos(a)) + y0);
      transformed_points2.append((x_t,y_t))


  cv2.line(immg, transformed_points1[0],transformed_points1[1], color=(0, 0, 0), thickness=2) 
  cv2.line(immg, transformed_points1[1],transformed_points1[2], color=(0, 0, 0), thickness=2) 
  cv2.line(immg, transformed_points1[2],transformed_points1[3], color=(0, 0, 0), thickness=2) 
  cv2.line(immg, transformed_points1[3],transformed_points1[0], color=(0, 0, 0), thickness=2) 

  cv2.line(immg, transformed_points2[orient[x][0]],transformed_points2[orient[x][1]], color=(0, 0, 0), thickness=2) 
  


cv2.imshow('image',img)



cv2.imshow('image',immg)

cv2.waitKey(0)
cv2.destroyAllWindows()