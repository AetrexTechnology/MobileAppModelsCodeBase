import cv2
import numpy as np
import cv2


def findCircles(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gray_blur = cv2.medianBlur(img_gray, 5)

    circles = cv2.HoughCircles(img_gray_blur, cv2.HOUGH_GRADIENT, 1.5, 10,
                               param1=100, param2=100, minRadius=25, maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)

            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)

            cv2.imshow('detected circles', image)
            print('Done')
    return


# Initialize webcam, cap is the object provided by VideoCapture
# It contains a boolean indicating if it was sucessful (ret)
# It also contains the images collected from the webcam (frame)
image_path = cv2.imread('/Users/vaneesh_k/PycharmProjects/Albie_ML/size-of-objects/images/floor_coin1.jpg')

frame = image_path
#    cv2.imshow('Our Live Sketcher', sketch(frame))
findCircles(frame)

# Release camera and close windows
cv2.destroyAllWindows()
