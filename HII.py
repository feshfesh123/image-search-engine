# descriptor het toan bo anh -> descList
from RGBHistogram import RGBHistogram
from ZernikeMoments import ZernikeMoments
import cv2
from matplotlib import pyplot as plt
import glob
import math
import pickle
from Searcher import Searcher
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# desc = RGBHistogram([8, 8, 8])
desc = RGBHistogram([4, 4, 4])


list_images = glob.glob("../data/flowers/*.jpg")
# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel


index = {}
i = 0
# use list_images to grab the image paths and loop over them
for imagePath in list_images:
    i+=1
    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind("/") + 1:]
    # load the image, describe it using our RGB histogram
    # descriptor, and update the index
    image = cv2.imread(imagePath)
    features = desc.describe(image)
    index[k] = [round(digit) for digit in features * 10e2]
    # index[k] = features
    # print(k, ":", index[k])
    # if (i >= 10):
    #     break

# tao 1 feature-values dua vao descList

desc_list = {}

# for i, feature in enumerate(index.values()): #load het cac features cua tat ca anh
#     for fi,value in enumerate(feature): # voi moi anh, duyet het 512 features,, tao inverted index
        # tao 1 dict
        # check xem fi (feature_value Fi) co ton tai, if not init, if yes add
        # sub_desc == Vij cua Features_value, sub_desc[gia tri tai Fi] = index cua tam anh
        # sub_desc = {}
        # desc_list[fi]
        # print(value)
# tao cac inverted list tuong ung voi cac feature-value

# 1 image -> 1 histogram -> 512 feature -> n value -> 1feature_value co m ids


feature_value_id = {}

for i, feature in enumerate(index.values()):
    for fi, value in enumerate(feature):
        if fi in feature_value_id:
            feature_item = feature_value_id[fi]
        else:
            feature_item = {}

        if value in feature_item:
            value_item = feature_item[value]
        else:
            value_item = []

        value_item.append(i)
        feature_item[value] = value_item
        feature_value_id[fi] = feature_item

# for i, value_ids in enumerate(feature_value_id.values()):
#     # for fi, value in enumerate(feature):
#     print("feature ", i, "-> value-ids :", value_ids)

def combine_list(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    list_2_items_not_in_list_1 = list(set_2 - set_1)
    combined_list = list_1 + list_2_items_not_in_list_1
    return combined_list


def draw_chart():
    x = []
    y = []
    # for feature in feature_value_id:
    #     for value in feature_value_id[feature].keys():
    #         x.append(feature)
    #         y.append(value)
    # # Basic chart
    # df = pd.DataFrame({'feature': x, 'value': y})
    # plt.plot('feature', 'value', data=df, linestyle='none', marker='o')
    # plt.show()

    df1 = {}
    df1['x'] = [i for i in range(1, 65)]

    for ih,histogram in enumerate(index.values()):
        df1['y' + str(ih)] = histogram
    df = pd.DataFrame(df1)
    # style
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

    # Add legend
    plt.legend(loc=2, ncol=2)
    plt.xlabel("feature")
    plt.ylabel("value")

    plt.show()


def chi2_distance(histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])
    # return the chi-squared distance
    return d

# draw_chart()
# exit()

query_path = "../data/flowers/image_0001.jpg"
query_image = cv2.imread(query_path)
query_histogram = desc.describe(query_image)
query_histogram = [round(digit) for digit in query_histogram * 10e2]


result_ids = []
results = []
for feature, value in enumerate(query_histogram):
    if feature in feature_value_id:
        if (value != 0):
            if (value in feature_value_id[feature]):
                ids = feature_value_id[feature][value]
                result_ids = combine_list(result_ids, ids)

print("result :", len(result_ids), ' /', i)

for i in result_ids:
    image = cv2.imread(list_images[i].replace('\\', '/'))
    features = desc.describe(image)
    histogram = [round(digit) for digit in features * 10e2]
    distance = chi2_distance(query_histogram, histogram)
    results.append({ 'index' :i, 'dist': distance})

results = sorted(results, key=lambda k: (k['dist']))

print(results)

for index in range(5):
    i = results[index]['index']
    list_images[i] = list_images[i].replace('\\', '/')
    print(list_images[i])
    image = cv2.imread(list_images[i])
    cv2.imshow(str(index), image)
    # image_path = "../data/flowers/image_" +  + ".jpg"
cv2.waitKey()
