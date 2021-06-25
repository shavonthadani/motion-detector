#motion_detector
#Shavon Thadani
#script takes the first frame of VideoCapture and compares it to the current frames.
#If there are differences, draw rectangle around it, showing the change in motion
#June 25, 2020

#imports
import cv2, time, pandas
from datetime import datetime
import os
#first frame to compare other frames to it
first_frame = None
#Access camera for video
video = cv2.VideoCapture(0)
#will record status if there is an object in the webcam
status_list = [None,None]
#record the times
times=[]
#count to adjust for ambient lighting
count = 0
#dataframe to organize times
df = pandas.DataFrame(columns=["Start","End"])

while True:
    #get the current frame from camera
    check, frame = video.read()
    #start off with status of no moving object
    status = 0
    #get a blurred gray scale version to compare first_frame with and have less noise
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    #count to wait for camera to adjust ambient lighting
    if count >100:
        #Executes first time through while loop. Associates the first frame of the video to the first_frame var
        if first_frame is None:
            first_frame=gray
            continue
        #delta_frame overlays the first_frame and the current frame and shows the differences
        delta_frame=cv2.absdiff(first_frame,gray)
        #assigning white pixels to values theshold values higher than 30
        thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
        #smooth threshold frame
        thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)
        #find the contours of the frame
        (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #draw rectangles around contours greater 10000 pixels
        for countour in cnts:
            if cv2.contourArea(countour) < 10000:
                continue
            #moving object in frame
            status = 1
            (x,y,w,h)=cv2.boundingRect(countour)
            cv2.rectangle(frame,(x, y),(x+w, y+h),(0,255,0), 3)
        #add the current status to status_list
        status_list.append(status)

        status_list = status_list[-2:]
        #time object comes into view. Get the time and append it to times list
        if status_list[-1]==1 and status_list[-2]==0:
            times.append(datetime.now())
        #time object leaves view. Get the time and append it to the times list
        if status_list[-1]==0 and status_list[-2]==1:
            times.append(datetime.now())
        #show the VideoCapture frames
        cv2.imshow('gray frame',gray)
        cv2.imshow('deltaframe', delta_frame)
        cv2.imshow('theshframe', thresh_frame)
        cv2.imshow("colorframe",frame)

        key=cv2.waitKey(1)
        #quit if q is pressed
        if key==ord('q'):
            if status==1:
                times.append(datetime.now())
            break
    #incement count
    count = count +1

print(status_list)
#iterate through times and count by 2, one for start, one for end
for i in range(0,len(times),2):
    #add start and end times to dataframe
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)
#write to csv file
df.to_csv(os.getcwd() + "/" + "Times.csv")
#stop video and destroy frames
video.release()
cv2.destroyAllWindows()
