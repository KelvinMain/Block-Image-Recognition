# ASSUMPTION: ALL IMAGES ARE CAPTURED FROM ROBOT HAND!!!
# TODO - FIND CONTOUR STORAGE TO ONLY TAKE CENTER SQUARE

import numpy as np
import cv2 as cv

def show_img(img_):
    def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv.resize(image, dim, interpolation=inter)

    imS = ResizeWithAspectRatio(img_, width=600)  # Resize image
    cv.imshow("output", imS)  # Show image

def getContours(edges, img_to_draw_the_contour_lines_on):
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE/NONE
    testing = []
    for cnt in contours:
        print(cnt)
        area = cv.contourArea(cnt)
        if area > 10:
            cv.drawContours(img_to_draw_the_contour_lines_on, cnt, -1, (255, 0, 255), 7)


def empty(_):
    pass


# SETUP for parameters in edge detecting
cv.namedWindow("Setting Parameters for edge detection")
cv.resizeWindow("Setting Parameters for edge detection", 640, 240)
cv.createTrackbar("Threshold 1", "Setting Parameters for edge detection", 0, 255, empty)
cv.createTrackbar("Threshold 2", "Setting Parameters for edge detection", 255, 255, empty)

path = 'block_detection_model_testing_photos/IM1.jpg'  # TODO - LOOPING FUNCTION FORALL PICS
img = cv.imread(path)

# Blur then Gray Image
Blurred_img = cv.GaussianBlur(img, (7, 7), 1)
Grayed_img = cv.cvtColor(Blurred_img, cv.COLOR_BGR2GRAY)

while True:
    # Canny edge detector
    threshold1 = cv.getTrackbarPos("Threshold 1", "Setting Parameters for edge detection")
    threshold2 = cv.getTrackbarPos("Threshold 2", "Setting Parameters for edge detection")
    Canny_img = cv.Canny(Grayed_img, threshold1, threshold2)

    kernel = np.ones((2, 2))
    Dil_img = cv.dilate(Canny_img, kernel, iterations=1)



    Contour_img = img.copy()
    getContours(Dil_img, Contour_img)

    show_img(Contour_img)
    show_img(Dil_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

'''gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray, 2, 3, 0.04)
dst = cv.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()'''
