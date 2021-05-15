import os

import cv2
import glob

img_dict = {}
for filename in glob.glob('C:/Users/zchen/PycharmProjects/opencv/googleNet/record/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_dict[filename.split("\\")[1]] = img
    #print(img_dict)
    print("loading image " + str(len(img_dict)))

path = 'C:/Users/zchen/PycharmProjects/opencv/googleNet'
out = cv2.VideoWriter(os.path.join(path , "fashion.avi"), cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(len(img_dict)):
    key = str(i) + ".jpg"
    out.write(img_dict[key])
    print("processing image " + str(i))
out.release()


