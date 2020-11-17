from RGBHistogram import RGBHistogram
import cv2
from matplotlib import pyplot as plt

test_image_path = '../data/flowers/image_0001.jpg'
test_image = cv2.imread(test_image_path)
# cv2.imshow("",test_image)

desc = RGBHistogram([8, 8, 8])

features = desc.describe(test_image)

plt.plot(features)
plt.show()
cv2.waitKey()