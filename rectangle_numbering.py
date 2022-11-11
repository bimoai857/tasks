import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('new_new.png')

imgray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours = " + str(len(contours)))


cv2.drawContours(img, contours, 3, (0, 255, 0), 1)
cv2.drawContours(img, contours, 6, (0, 255, 0), 1)
cv2.drawContours(img, contours, 9, (0, 255, 0), 1)
cv2.drawContours(img, contours, 12, (0, 255, 0), 1)

box2=[]

for i in range(3,13,3):
  box2.append(np.int0(cv2.boxPoints(cv2.minAreaRect(contours[i]))))

dist1=np.linalg.norm(np.array(box2[0][1])-(box2[0][2]))
dist2=np.linalg.norm(np.array(box2[1][0])-(box2[1][1]))
dist3=np.linalg.norm(np.array(box2[2][0])-(box2[2][3]))
dist4=np.linalg.norm(np.array(box2[3][0])-(box2[3][1]))

cv2.imshow('image',img)



dic={dist1:'br',dist2:'bl',dist3:'tl',dist4:'tr'}
lst=[dist1,dist2,dist3,dist4]
lst.sort()

print(dic)
print(lst)

x=50
y=185
org = (x,y)
  
font = cv2.FONT_HERSHEY_SIMPLEX  

fontScale = 1

color = (0, 0, 255)
  

thickness = 2

text=0
for i in lst:

  text=text+1

  if(dic[i]=='br'):
      cv2.putText(img, str(text), (700,400), font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)

  
  elif(dic[i]=='bl'):
      cv2.putText(img, str(text), (50,400), font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)

  elif(dic[i]=='tl'):
      cv2.putText(img, str(text), (50,185), font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)
      
  else:
      cv2.putText(img, str(text), (700,185), font, fontScale, 
                 color, thickness, cv2.LINE_AA, False)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
