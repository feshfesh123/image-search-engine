from RGBHistogram import RGBHistogram
from ZernikeMoments import ZernikeMoments
import cv2
from matplotlib import pyplot as plt

test_image_path = '../data/flowers/image_0030.jpg'
test_image = cv2.imread(test_image_path)

# cv2.imshow("",test_image)

#color = RGBHistogram([8, 8, 8])

#features = color.describe(test_image)
#plt.plot(features)
#plt.show()

shape = ZernikeMoments(21)

features = shape.describe(test_image)

cv2.imshow("edge", features)

cv2.waitKey()