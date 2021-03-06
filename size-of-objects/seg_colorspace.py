import numpy as np
import cv2


# this code hsv used to properly mask the coin
def findCircles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # For color space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # generate edges using canny edge detector

    HSVLOW = np.array([0, 0, 0])
    HSVHIGH = np.array([179, 31, 225])

    # apply the range on a mask
    # mask = cv2.inRange(gray, HSVLOW, HSVHIGH)
    # maskedFrame = cv2.bitwise_and(image, image, mask=mask)

    img_gray_blur = cv2.medianBlur(gray, 5)
    # img_gray_blur = cv2.cvtColor(img_gray_blur, cv2.COLOR_BGR2GRAY)
    # img_gray_blur  = cv2.Canny(img_gray_blur,threshold1=10, threshold2=50)
    circles = cv2.HoughCircles(img_gray_blur, cv2.HOUGH_GRADIENT, 1.2, 20,
                               param1=25, param2=50, minRadius=10, maxRadius=40)

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
image_path = cv2.imread('../size-of-objects/images/cropped/kumar.jpg')

frame = image_path
#    cv2.imshow('Our Live Sketcher', sketch(frame))
findCircles(frame)
print('----Final-----')
# Release camera and close windows
cv2.destroyAllWindows()
