import numpy as np
import cv2

test_image_path = '../data/flowers/image_0030.jpg'
image_vec = cv2.imread(test_image_path, 1)
g_blurred = cv2.GaussianBlur(image_vec, (5, 5), 0)

blurred_float = g_blurred.astype(np.float32) / 255.0
edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
edges = edgeDetector.detectEdges(blurred_float) * 255.0
cv2.imwrite('edge-raw.jpg', edges)

def SaltPepperNoise(edgeImg):

    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)

edges_ = np.asarray(edges, np.uint8)
SaltPepperNoise(edges_)
cv2.imshow('edge.jpg', edges_)


def findSignificantContour(edgeImg):
    image, contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

contour = findSignificantContour(edges_)
# Draw the contour on the original image
contourImg = np.copy(image_vec)
cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
cv2.imshow('contour.jpg', contourImg)
