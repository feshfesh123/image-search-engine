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
from Feature_Extract_Resnet18 import Resnet18Feature


modelPath = 'models/my_feature_Resnet18_descriptor_.pth'
desc = Resnet18Feature(modelPath)

list_images = glob.glob("../data/flowers/*.jpg")

index = {}
i = 0
for imagePath in list_images:
    i+=1
    k = imagePath[imagePath.rfind("/") + 1:]
    img = cv2.imread(imagePath)
    features = desc.describe(img)
    index[k] = features
    print(features)
    quit()
    # index[k] = features
    # print(k, ":", index[k])
    # if (i >= 10):
    #     break