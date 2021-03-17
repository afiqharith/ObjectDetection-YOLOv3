__author__ = "Afiq Harith"
__email__ = "afiqharith05@gmail.com"
__date__ = "17 Oct 2020"
__status__ = "Build-pass"

import cv2
import numpy as np
import math
import os
from setup.model import Model
from setup.config import *

# Load image
IMAGEPATH = os.path.join(os.getcwd(), IMAGEFOLDER, IMAGENAME)

class ObjectDetection:
    
    def __init__(self, source = IMAGEPATH, START = True):

        self.image = cv2.imread(source)
        self.model = Model(UTILS, MODELPATH, WEIGHTS, CFG, COCONAMES)
        if START == True: self.main()
    
    def draw_detection_box(self, frame, xmn, ymn, xmx, ymx, color):
        cv2.rectangle(frame, (xmn, ymn), (xmx, ymx), color, 2)
        
    def main(self):

        while True:
            net, layerNames, classes = net, layerNames, classes = self.model.predict()

            frame_resized = cv2.resize(self.image, (416,416)) # resize frame for prediction       

            frame_rgb = cv2.cvtColor(self.image, cv2.IMREAD_COLOR)
            height, width, channels = self.image.shape

            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            layerOutputs = net.forward(layerNames)

            classIDs = []
            confidences = []
            boxes = []
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

            # apply non-max suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    xmin = x
                    ymin = y
                    xmax = (x + w)
                    ymax = (y + h)

                    self.draw_detection_box(self.image, xmin, ymin, xmax, ymax, BLUE)
                    label = f"{classes[classIDs[i]]} {'%.2f' % confidences[i]}"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                    y1label = max(ymin, labelSize[1])
                    cv2.rectangle(self.image, (xmin, y1label - labelSize[1]),(xmin + labelSize[0], ymin + baseLine), WHITE, cv2.FILLED)
                    cv2.putText(self.image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1,cv2.LINE_AA)

            cv2.imshow("YOLO Object Detection", self.image)
            '''
            :Pressed 'Q': Exit application
            '''
            if cv2.waitKey(0) % 256 == ord('q'):
                break
        

if __name__ == '__main__':

    print('Press \'Q\' to exit.')
    ObjectDetection(IMAGEPATH)
    cv2.destroyAllWindows()