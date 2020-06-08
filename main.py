import urllib.request
from cv2 import cv2
import numpy as np

# IPv4 url from IP Webcam
url = "https://192.168.1.140:8080"


def getImage(img_url):
    img_arr = np.array(
        bytearray(urllib.request.urlopen(img_url).read()),
        dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img


def getEdges(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 10, 100)
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)
    img_edges = cv2.erode(img_dial, kernel, iterations=1)

    return img_edges


def getApprox(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        area_ratio = cv2.contourArea(contour) / (img.shape[0] * img.shape[1])

        if len(approx) == 4 and area_ratio > 0.3 and area_ratio < 0.8:
            print(area_ratio)

            return approx, contour

    return None, None


def getWarp(img, approx):
    pass


while True:
    img = getImage(url + "/shot.jpg")

    img_edges = getEdges(img)
    img_contours = img.copy()

    approx, contour = getApprox(img_edges)

    getWarp(img, approx)

    if approx is not None:
        cv2.drawContours(img_contours, [approx], -1, (0, 255, 0), 10)
        cv2.drawContours(img_contours, approx, -1, (255, 0, 0), 50)

    cv2.imshow("Image", img_contours)
    cv2.imshow("Edged", img_edges)

    if contour is not None:
        pass

    if cv2.waitKey(1) & 0xff == 27:
        break
