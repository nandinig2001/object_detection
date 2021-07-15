import cv2
import numpy as np 

# YOLO is a clever neural network for doing object detection in real-time

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []

# COCO is a large detection dataset from Microsoft with 80 object categories.
with open('coco.names','r') as f:
    classes = f.read().splitlines()

# cap contains the video to be captured
cap = cv2.VideoCapture('street.mp4')

while True:
    if cap.isOpened():
        _, img = cap.read()
    height, width, _ = img.shape

# construct a blob from the input frame and then perform a forward pass for the bounding boxes

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutput =net.forward(output_layers_names)

# boxes : Bounding boxes around the object.
# confidences : The confidence value that YOLO assigns to an object. Lower confidence values indicates low accuracy of detection
# class ID : The detected object class label.

    boxes = []
    confidences = []
    class_ids = []

    for output in layersOutput:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

# Only considering objects greater than 0.5 confidence for accuracy in the model
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255, size=(len(boxes),3))

    for i in indexes.flatten():
    
# formation of the boxes around the objects in frame

        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label + " " + confidence, (x,y+20), font, 2, (255,255,255), 2)


# Specifying the formation of the window

    cv2.imshow('Object Detection', img)

# command to close the window

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
