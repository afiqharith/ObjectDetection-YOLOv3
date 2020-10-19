__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "18 Oct 2020"
__status__ = "Build-pass"

import cv2
import numpy as np
import math
import os
from setup.model import dataFromModel
from setup.config import *
from setup.tracker import CentroidTracker

# Load video
VIDEOPATH = os.path.join(os.getcwd(), VIDEOFOLDER, VIDEONAME)

class ObjectDetection:

    def __init__(self, VIDEOPATH, CAMERA, START = True):

        if CAMERA == True:
            self.video = cv2.VideoCapture(0)
        else:
            self.video = cv2.VideoCapture(VIDEOPATH)

        if START == True:
            self.main()
    
    def draw_detection_box(self, frame, xmn, ymn, xmx, ymx, color):
        cv2.rectangle(frame, (xmn, ymn), (xmx, ymx), color, 2)
        
    def main(self):
        net, layerNames, classes = dataFromModel.get(MODELPATH, WEIGHTS, CFG, COCONAMES)

        while (self.video.isOpened()):
            self.ret, self.frame = self.video.read() 

            if self.ret:
                frame_resized = cv2.resize(self.frame, (416,416)) # resize frame for prediction       
            else:
                break
            frame_rgb = cv2.cvtColor(self.frame, cv2.IMREAD_COLOR)
            height, width, channels = self.frame.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            layerOutputs = net.forward(layerNames)

            classIDs = []
            confidences = []
            boxes = []

            # For object tracker from pyimagesarch
            rects = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    #if prediction is 50%
                    if confidence > CONFIDENCE:

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
                        classIDs.append(classID)

                        # For object tracker from pyimagesarch
                        box = np.array([x, y, x+w, y+h])
                        rects.append(box.astype("int"))
                        
            # apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

            # For object tracker from pyimagesarch
            objects = CentroidTracker().update(rects)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmin = x
                    ymin = y
                    xmax = (x + w)
                    ymax = (y + h)

                    self.draw_detection_box(self.frame, xmin, ymin, xmax, ymax, BLUE)
                    label = f"{classes[classIDs[i]]} {'%.2f' % confidences[i]}"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    y1label = max(ymin, labelSize[1])
                    cv2.rectangle(self.frame, (xmin, y1label - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(self.frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1,cv2.LINE_AA)
                    
                    # For object tracker from pyimagesarch
                    if TRACKER == True:
                        for (objectID, centroid) in objects.items():
                            text = f'{objectID}'
                            cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1,cv2.LINE_AA)
                            cv2.circle(self.frame, (centroid[0], centroid[1]), 4, GREY, -1, cv2.LINE_AA)


            cv2.imshow("YOLO Object Detection", self.frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        self.video.release()

if __name__ == '__main__':

    ObjectDetection(VIDEOPATH, CAMERA)
    cv2.destroyAllWindows()