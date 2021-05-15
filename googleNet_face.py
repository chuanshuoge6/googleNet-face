import os

import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="googleNet\\deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",
                default="googleNet\\res10_300x300_ssd_iter_140000.caffemodel",
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture("assets/fashion.mp4")

j = 0
while True:
    ret, frame = cap.read()
    w = int(cap.get(3))
    h = int(cap.get(4))

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    path = 'C:/Users/zchen/PycharmProjects/opencv/googleNet/record'
    name = str(j) + ".jpg"
    cv2.imwrite(os.path.join(path , name), frame)
    j += 1

    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('p'):
        cv2.waitKey(-1)  # wait until any key is pressed

cap.release()
cv2.destroyAllWindows()

