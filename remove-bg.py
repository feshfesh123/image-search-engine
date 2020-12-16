
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
query_path = "../data/flowers/image_1281.jpg"

img = cv2.imread(query_path)

# converting image into its hsv form
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), cmap='hsv', marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# selecting the color range to be extracted
lower_green = np.array([299.1367, 100.0000, 54.5098])  # lowest range
upper_green = np.array([262.5000, 45.4976, 82.7451])  # highest range


# creating mask for image segmentation
# mask = cv2.inRange(hsv, lower_green, upper_green)

mask = cv2.inRange(hsv, upper_green, lower_green)


# extracting the foreground from the image
fg = cv2.bitwise_and(img, img, mask=mask)

# saving the extracted image
cv2.imshow('remove-bg', fg)
cv2.waitKey()
