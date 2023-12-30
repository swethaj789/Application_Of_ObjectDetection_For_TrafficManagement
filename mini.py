from ultralytics import YOLO
import numpy as np
import cv2
import math
import cvzone
from sort import*
#from tracker import*
#cap.set(3,1280)
#cap.set(4,720)

frameWidth=1000
frameHeight=500

cap = cv2.VideoCapture("/home/student/Downloads/pexels-hervé-piglowski-5649316 (1080p).mp4")

model = YOLO("yolov8n.pt")

classNames=[]
classFile="/home/student/Downloads/labels.txt"
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
   
   
#Tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)


#limits=[100,800,1000,800]

limits=[500,800,980,800]
#limits=[500,800,1500,800]
limits1=[1500,850,1050,850]
#limits1=[1500,750,1050,750]

directions={}

totalcount_forward=[]
totalcount_backward=[]

while True:
    succes, img=cap.read()
    results=model(img,stream=True)
   
    detections=np.empty((0, 5))
   
   
    for r in results:
        boxes=r.boxes
        for box in boxes:
           
            #bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
           
            #x1,y1,w,h=box.xywh[0]
            #x1,y1,x2,y2=int(x1),int(y1),int(w),int(h)
            #cv2.rectangle(img,(x1,y1),(x1+w,y2+h),(255,0,255),3)
            w,h=x2-x1,y2-y1
            #print(x1,y1,w,h)
           
           
           
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1))
            #classNames
            cls=int(box.cls[0])
            currentclass=classNames[cls]
           
            if currentclass == "car" or currentclass == "truck" or currentclass == "motorbike" or currentclass == "bus" :#and conf>0.3:
                               
                cvzone.putTextRect(img,f'{currentclass} {conf}',(max(0,x1),max(35,y1)),scale=0.7,thickness=1,offset=3)                  
               
                cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
               
                currentArray=np.array([x1,y1,x2,y2,conf])
               
                detections=np.vstack((detections,currentArray))
           
   
    resultsTracker=tracker.update(detections)
   
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    cv2.line(img,(limits1[0],limits1[1]),(limits1[2],limits1[3]),(0,0,255),5)
   
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
       
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=10)
       
        cx,cy=x1+w//2,y1+h//2
        cx,cy=(x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
       
        #if the object's id is not in the direction dictionary,add it with initial direction
        if id not in directions:
            directions[id]="unknown"
       
        #check if the object has crossed the line (forward or backward)
        if limits[0]<cx<limits[2] and limits[1]-5<cy<limits[1]+5 or limits[0]<cx<limits[2] and limits[1]-1<cy<limits[1]+1:
            #if the object is crossing the line from left to right,mark it as forward
            if directions[id]=="unknown" or directions[id]=="backward":
                directions[id]="forward"
                if totalcount_forward.count(id)==0:
                    totalcount_forward.append(id)
                    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
             
           
            #if the object is crossing the line from right to left,mark it as backward
            elif directions[id]=="forward":
                directions[id]="backward"
                if totalcount_backward.count(id)==0:
                    #totalcount_forward.remove(id)
                    totalcount_backward.append(id)
                    cv2.line(img,(limits1[0],limits1[1]),(limits1[2],limits1[3]),(255,0,0),5)    
   
   
    cvzone.putTextRect(img,f'vehicles forwarding:{len(totalcount_forward)}',(50,50))
    cvzone.putTextRect(img,f'vehicles backwarding:{len(totalcount_backward)}',(1300,50))
    frame=cv2.resize(img,(frameWidth,frameHeight))
    cv2.imshow("image",frame)
    cv2.waitKey(1)

