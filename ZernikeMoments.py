# import the necessary packages
import imutils
import mahotas
import cv2
import numpy as np

class ZernikeMoments:
	def __init__(self, radius):
		# store the size of the radius that will be
		# used when computing moments
		self.radius = radius

	def preprocess(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# pad the image with extra white pixels to ensure the
		# edges of the pokemon are not up against the borders
		# of the image
		image = cv2.copyMakeBorder(image, 15, 15, 15, 15,
								   cv2.BORDER_CONSTANT, value=255)
		# invert the image and threshold it
		thresh = cv2.bitwise_not(image)
		thresh[thresh > 0] = 255

		# initialize the outline image, find the outermost
		# contours (the outline) of the pokemone, then draw
		# it
		outline = np.zeros(image.shape, dtype="uint8")
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
		cv2.drawContours(outline, [cnts], -1, 255, -1)
		return outline

	def describe(self, image):
		outline = self.preprocess(image)
		cv2.imshow("", outline)
		# return the Zernike moments for the image
		return mahotas.features.zernike_moments(outline, self.radius)