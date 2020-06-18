import numpy as np
import cv2


def findCircles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # For color space
    # make array for final values

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(gray)
    # generate edges using canny edge detector
    # edges = canny(gray, sigma=3, low_threshold=10, high_threshold=50)

    # For color space
    # make array for final values
    HSVLOW = np.array([0, 0, 0])
    HSVHIGH = np.array([180, 255, 255])

    # apply the range on a mask
    mask = cv2.inRange(gray, HSVLOW, HSVHIGH)
    # maskedFrame = cv2.bitwise_and(image, image, mask=mask)

    img_gray_blur = cv2.medianBlur(mask, 5)
    # img_gray_blur = cv2.cvtColor(img_gray_blur, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(img_gray_blur, cv2.HOUGH_GRADIENT, 1.2, 20,
                               param1=10, param2=20, minRadius=10, maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)

            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            result = cv2.resize(image, (800, 800))
            cv2.imshow('image', result)
            key = cv2.waitKey(0)  # pauses for 3 seconds before fetching next image
            if key == 27:  # if ESC is pressed, exit loop
                cv2.destroyAllWindows()
                break
            print('Done')
    return


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
image_path = cv2.imread(
    '/size-of-objects/images/cropped/balcony_1.jpg')

frame = image_path
#    cv2.imshow('Our Live Sketcher', sketch(frame))
findCircles(frame)
print('----Final-----')
# Release camera and close windows
cv2.destroyAllWindows()
