import glob
import os

import numpy as np
import cv2

directory = '../regression_modeldata1'
subdirs = os.listdir(directory)

rgb_weights = [0.2989, 0.5870, 0.1140]


def createMask(imagePath):
    image = cv2.imread(imagePath)
    boundaries = [
        # ([100, 0, 0], [255, 70, 79]),
        ([0, 0, 0], [255, 0, 20])
    ]
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        max_y = getPixelLocation(output) + 10
        crop_image = image[max_y - image.shape[1]:max_y, :, :]
        cv2.imwrite(
            '../size-of-objects/images/cropped/kumar.jpg',
            crop_image)
        # cv2.imshow("images", np.hstack([image, output]))
        # cv2.waitKey(0)
        # cv2.imshow("cropped", crop_image)
        # cv2.waitKey(0)


def getPixelLocation(mask):
    grayscale_image = np.dot(mask[..., :3], rgb_weights)
    output2 = np.where(grayscale_image != 0)
    max_y = output2[0].max()
    return max_y

# for i in range(len(subdirs)):
#     dir2pull = directory + '/' + subdirs[i] + '/'
#     for file in glob.glob(dir2pull + '*.jpg'):
#         createMask(file)
imagePath = '../size-of-objects/images/kumar.jpg'

createMask(imagePath)
print('Done')
