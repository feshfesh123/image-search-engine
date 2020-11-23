from RGBHistogram import RGBHistogram
from ZernikeMoments import ZernikeMoments
import cv2
from matplotlib import pyplot as plt
import glob
import pickle
from Searcher import Searcher

desc = RGBHistogram([8, 8, 8])


test_image_path = '../data/flowers/image_0030.jpg'
query_image = cv2.imread(test_image_path)
query_feature = desc.describe(query_image)
print(query_feature)
quit()

# load the index and initialize our searcher
index = pickle.load(open("Histogram_only_index.cpickle", "rb"))
# print(index.items())
searcher = Searcher(index)
results = searcher.search(query_feature)
print(results)
for i in range(0, 5):
	ret_path = '../data/' + results[i][1].replace('\\','/')
	img = cv2.imread(ret_path)
	cv2.imshow(str(i), img)

cv2.waitKey()
exit()

list_images = glob.glob("../data/flowers/*.jpg")
# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel


index = {}

# use list_images to grab the image paths and loop over them
for imagePath in list_images:
	# extract our unique image ID (i.e. the filename)
	k = imagePath[imagePath.rfind("/") + 1:]
	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = desc.describe(image)
	index[k] = features

# we are now done indexing our image -- now we can write our
# index to disk
f = open("Histogram_only_index.cpickle", "wb")
f.write(pickle.dumps(index))
f.close()

print("[INFO] done...indexed {} images".format(len(index)))


