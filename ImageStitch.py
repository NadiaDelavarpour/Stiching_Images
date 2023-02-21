#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import glob
import imutils
import os

dir_path = 'C:\\Users\\nadia\\Downloads\\stitchimage'
# Use os.path.join() to build the file path for each image
image_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.jpg')]
images = []



for image in image_paths:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


imageStitcher = cv2.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:
    # Use os.path.join() to build the output file path based on the directory path and the desired output file name
    output_path = os.path.join(dir_path, 'stitchedOutput.png')

    cv2.imwrite(output_path, stitched_img)
    cv2.imshow("Stitched Img", stitched_img)
    cv2.waitKey(0)




    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

    cv2.imshow("Threshold Image", thresh_img)
    cv2.waitKey(0)

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)


    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    cv2.imshow("minRectangle Image", minRectangle)
    cv2.waitKey(0)

    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]
    output_path1 = os.path.join(dir_path, "stitchedOutputProcessed.png")

    cv2.imwrite(output_path1, stitched_img)

    cv2.imshow("Stitched Image Processed", stitched_img)

    cv2.waitKey(0)



else:
    print("Images could not be stitched!")
    print("Likely not enough keypoints being detected!")


# In[ ]:





# In[ ]:




