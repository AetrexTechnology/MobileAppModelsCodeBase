import numpy as np
import cv2

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


# load the image, clone it for output, and then convert it to grayscale
image_file = '/Users/vaneesh_k/PycharmProjects/Albie_ML/size-of-objects/images/cropped/balcony_coin1_cropped.jpg'
image = cv2.imread(image_file)
# image = cv2.resize(image, (800, 800))
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# generate edges using canny edge detector
edges = canny(gray, sigma=3, low_threshold=10, high_threshold=50)
cv_edges = img_as_ubyte(edges)
cv2.imshow("cv_edges", cv_edges)

# compute hough circel transformation with the range of [20, 70]
hough_radii = np.arange(20, 70)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

# draw the circle in the output image, then draw a rectangle
# corresponding to the center of the circle
for y, x, r in zip(cy, cx, radii):
    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# show the output image
result = np.hstack([cv2.cvtColor(cv_edges, cv2.COLOR_GRAY2RGB), output])
output = cv2.resize(result,(800,800))
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.imshow("output", output)
cv2.waitKey(0)

# save result image

print('Done')
# cv2.imwrite("result.png", result)
