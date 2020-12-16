from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.layers import Flatten
# from keras.models import Model
import glob

from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

from keras.applications import VGG16
from keras.models import Model


def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d


vgg = VGG16(
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000)

OUTPUT_LAYERS = [
    'block1_pool',
    'block2_pool',
    'block3_pool',
    'block4_pool',
    'block5_pool'
    ]

outputs = [vgg.get_layer(l).output for l in OUTPUT_LAYERS]

feature_extractor = Model(inputs=vgg.input, outputs=outputs)
feature_extractor.summary()

list_images = glob.glob("../data/flowers/*.jpg")

index = {}
i = 0
# use list_images to grab the image paths and loop over them
for imagePath in list_images:
    i+=1
    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind("/") + 1:]
    # load the image, describe it using our RGB histogram
    # descriptor, and update the index
    img = image.load_img(imagePath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = vgg.predict(img_data)[0]
    # index[k] = [round(digit) for digit in features * 10e2]
    index[k] = features
    # print(k, ":", index[k])
    if (i >= 100):
         break


#query test
query_path = "../data/flowers/image_0103.jpg"
query_img = image.load_img(query_path, target_size=(224, 224))
query_data = image.img_to_array(query_img)
query_data = np.expand_dims(query_data, axis=0)
query_data = preprocess_input(query_data)
query_features = vgg.predict(query_data)[0]

#distance
results = []
for i in index:

    distance = chi2_distance(index[i], query_features)
    results.append({ 'index' :i, 'dist': distance})

results = sorted(results, key=lambda k: (k['dist']))

for index, i in enumerate(results):
    if index == 5:
        exit()
    print(i)


