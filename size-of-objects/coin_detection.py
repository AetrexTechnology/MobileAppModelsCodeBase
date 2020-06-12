import numpy as np
import argparse
import cv2

# load the image, clone it for output, and then convert it to grayscale
image_file = '/Users/vaneesh_k/PycharmProjects/Albie_ML/size-of-objects/images/coin.png'
image = cv2.imread(image_file)
image = cv2.resize(image, (800, 800))
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# mask =

cv2.imshow('mask',gray)
# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 4.0, 100, 100.0, 30.0, 20, 200)
print('Found some circles')
# ensure at least some circles were found
if circles is not None:
    print('filterd circles')
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # show the output image
    cv2.imshow("output", np.hstack([image, output]))
    cv2.waitKey(0)
