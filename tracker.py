from sort import *

sort_tracker = Sort()
#
#
# # update SORT

import cv2
import threading
import time, timeit
import detector

url = 'rtsp://admin:parol12345@192.168.4.220:554/cam/realmonitor?channel=1&subtype=0'
# url = 'rtsp://admin:Aa123456@10.4.38.12:554/StreamingSetting?version=1.0&action=getRTSPStream&ChannelID=1&ChannelName=Channel1'
# url = 'traffic2.mp4'
getFrame = True
frame = None
cv2.namedWindow("w", cv2.WINDOW_KEEPRATIO)
detector_ = detector.Detector()


def drawLimit():
    global frame
    # w,h = frame.shape[:2]
    # cv2.line(frame,(0,h),(w,h),(0,0,255),10)
    # cv2.line(frame, (0, 231), (w, 231), (0, 0, 255), 10)


def detectCar():
    global frame, detector_
    timer = cv2.getTickCount()
    frame = detector_.drawBoxes(frame)
    # # print(dets)
    # for det in dets:
    #     track_bbox_ids = sort_tracker.update(np.array(dets))
    #     print(track_bbox_ids)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    cv2.putText(frame, "FPS : " + str(int(fps)), (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def getCap():
    global getFrame, frame
    cap = cv2.VideoCapture(url)
    while True:
        if cap.grab():
            if getFrame:
                r, frame = cap.retrieve()
                # frame = cv2.resize(frame,(1000,500))
                # drawLimit()
                getFrame = False


def showCap():
    global getFrame, frame
    while True:
        if not getFrame:
            detectCar()
            cv2.imshow('w', frame)
            getFrame = True


x1 = threading.Thread(target=getCap)
x1.start()
time.sleep(2)
x2 = threading.Thread(target=showCap)
x2.start()

cv2.waitKey(0)
