from imutils import paths
import numpy as np
import cv2
from os.path import dirname, abspath

# load the class labels from disk
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
                               "bvlc_googlenet.caffemodel")

# grab the paths to the input images
path = dirname(dirname(abspath(__file__))) + "\\assets\\googleNet"
imagePaths = sorted(list(paths.list_images(path)))
# print(path, imagePaths)

# (1) load the first image from disk, (2) pre-process it by resizing
# it to 224x224 pixels, and (3) construct a blob that can be passed
# through the pre-trained network
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))
blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("First Blob: {}".format(blob.shape))

# set the input to the pre-trained deep learning network and obtain
# the output predicted probabilities for each of the 1,000 ImageNet
# classes
net.setInput(blob)
preds = net.forward()

# sort the probabilities (in descending) order, grab the index of the
# top predicted label, and draw it on the input image
idx = np.argsort(preds[0])[::-1][0:4]
for j, label_id in enumerate(idx):
    text = "Label: {}, {:.2f}%".format(classes[label_id], preds[0][label_id] * 100)
    cv2.putText(image, text, (5, 25 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

# initialize the list of images we'll be passing through the network
images = []

# loop over the input images (excluding the first one since we
# already classified it), pre-process each image, and update the
# `images` list
for p in imagePaths[1:]:
    image = cv2.imread(p)
    image = cv2.resize(image, (224, 224))
    images.append(image)

# convert the images list into an OpenCV-compatible blob
blob = cv2.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(blob.shape))

# set the input to our pre-trained network and obtain the output
# class label predictions
net.setInput(blob)
preds = net.forward()

# loop over the input images
for (i, p) in enumerate(imagePaths[1:]):
    # load the image from disk
    image = cv2.imread(p)

    # find the top class label from the `preds` list and draw it on
    # the image
    idx = np.argsort(preds[i])[::-1][0:4]
    for j, label_id in enumerate(idx):
        text = "Label: {}, {:.2f}%".format(classes[label_id], preds[i][label_id] * 100)
        cv2.putText(image, text, (5, 25 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # display the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
