import numpy as np
import cv2

#define image used
path = ""
img = cv2.imread(path)
if img is None:
    print('Invalid path')

#define objects we are searching for (see training_classifications.txt)
classifications_allowed = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

#initialize model
dnn = cv2.dnn.readNet("dnn/yolov4.weights", "dnn/yolov4.cfg")
model = cv2.dnn_DetectionModel(dnn)
model.setInputParams(size=(832, 832), scale=1 / 255)

#detect traffic
def detect_traffic(img):
    traffic = []
    class_ids, confidences, boxes = model.detect(img, nmsThreshold=0.4)
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        if confidence > 0.5:
            if class_id in list(classifications_allowed.keys()):
                traffic.append([class_id, box])
    return traffic

traffic = detect_traffic(img)

#display boxes and text to image
for object in traffic:
        x, y, w, h = object[1]
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
        cv2.putText(img, classifications_allowed[object[0]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 0, 180), 3)
        traffic_count = len(traffic)
        cv2.putText(img, "Traffic Objects: " + str(traffic_count), (20, 50), 0, 2, (100, 200, 0), 2)

cv2.imshow("traffic", img)
cv2.waitKey()