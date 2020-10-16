__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "08 Oct 2020"
__status__ = "Development-OOP"

import cv2
import numpy as np
import math
import os
from setup.LoadModel import LoadModel
from setup import config as cfg

GREEN   = cfg.GREEN
RED     = cfg.RED
YELLOW  = cfg.YELLOW
WHITE   = cfg.WHITE
ORANGE  = cfg.ORANGE
BLUE    = cfg.BLUE
GREY    = cfg.GREY

class SODV:
    def __init__(self,videoSRC, STAT = True):
        self.video = cv2.VideoCapture(videoSRC)
        if STAT == True:
            self.main()
    
    def draw_detection_box(self,frame,x1,y1,x2,y2,color):
        cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
        
    def main(self):
        net, output_layers, classes = LoadModel.get()

        while True:
            ret, frame = self.video.read() 

            if ret:
                frame_resized = cv2.resize(frame, (416,416)) # resize frame for prediction       
            else:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            height, width, channels = frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    #if prediction is 50% and class id is 0 which is 'person'
                    if confidence > 0.5:
                        
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmin = x
                    ymin = y
                    xmax = (x + w)
                    ymax = (y + h)

                    self.draw_detection_box(frame,xmin,ymin,xmax,ymax,BLUE)
                    label = classes[class_ids[i]]
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    y1label = max(ymin, labelSize[1])
                    cv2.rectangle(frame, (xmin, y1label - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1,cv2.LINE_AA)

            cv2.imshow("YOLO Object Detection", frame)

            if cv2.waitKey(1) >= 0:  
                break

        self.video.release()

if __name__ == '__main__':

    # Load video
    videoName = "TownCentre.mp4"
    videosPath = "videos/"
    videoSRC = os.path.join(os.getcwd(), videosPath, videoName)

    SODV(videoSRC)
    cv2.destroyAllWindows()