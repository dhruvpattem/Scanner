import urllib.request
from cv2 import cv2
import numpy as np

# IPv4 url from IP Webcam
url = "http://192.168._.__:8080"

# Device Webcam
cap = cv2.VideoCapture(0)

ipwebcam = False if '_' in url else True


def getImage(img_url):
    if ipwebcam:
        img_arr = np.array(
            bytearray(urllib.request.urlopen(img_url).read()),
            dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        img = cap.read()[1]

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

            return approx


def getWarp(img, approx):
    height = img.shape[0]
    width = int(height * (210/297))
    pts1 = np.float32(reorderPoints(approx))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (width, height))

    return img_warped


def getScan(img_warped):
    img_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    img_sharp = cv2.GaussianBlur(img_gray, (0, 0), 3)
    img_sharp = cv2.addWeighted(img_gray, 1.5, img_sharp, -0.5, 0)

    img_scanned = cv2.adaptiveThreshold(
        img_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

    return img_scanned


def reorderPoints(pts):
    pts_reshape = pts.reshape((4, 2))
    pts_new = np.zeros((4, 1, 2), np.int32)

    add = pts_reshape.sum(1)
    diff = np.diff(pts_reshape, axis=1)

    pts_new[0] = pts_reshape[np.argmin(add)]
    pts_new[3] = pts_reshape[np.argmax(add)]

    pts_new[1] = pts_reshape[np.argmin(diff)]
    pts_new[2] = pts_reshape[np.argmax(diff)]

    return pts_new


while True:
    img = getImage(url + "/shot.jpg")

    img_edges = getEdges(img)
    img_contours = img.copy()

    approx = getApprox(img_edges)

    cv2.imshow("Image", img)
    cv2.imshow("Image2", img_edges)

    if approx is not None:
        cv2.drawContours(img_contours, [approx], -1, (0, 255, 0), 10)
        cv2.drawContours(img_contours, approx, -1, (255, 0, 0), 50)

        img_warped = getWarp(img, approx)
        cv2.imshow("Contours", img_contours)
        cv2.imshow("Scanned", getScan(img_warped))

    if cv2.waitKey(1) & 0xff == 27:
        break
