# descriptor het toan bo anh -> descList
from RGBHistogram import RGBHistogram
from ZernikeMoments import ZernikeMoments
import cv2
from matplotlib import pyplot as plt
import glob
import pickle
from Searcher import Searcher

desc = RGBHistogram([8, 8, 8])

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
    index[k] = [round(digit) for digit in features * 10e5]


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


query_path = "../data/flowers/image_0001.jpg"
query_image = cv2.imread(query_path)
query_histogram = desc.describe(query_image)

def combine_list(list_1, list_2):
    set_1 = set(list_1)
    set_2 = set(list_2)
    list_2_items_not_in_list_1 = list(set_2 - set_1)
    combined_list = list_1 + list_2_items_not_in_list_1
    return combined_list

result_ids = []
for feature, value in enumerate(query_histogram):
    if feature in feature_value_id:
        if (value in feature_value_id[feature]) and (value != 0):
            ids = feature_value_id[feature][value]
            result_ids = combine_list(result_ids, ids)

print("data length :", i)
print("result :", len(result_ids))