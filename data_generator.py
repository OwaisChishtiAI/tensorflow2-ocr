# USAGE
# python train_ocr_model.py --az a_z_handwritten_data.csv --model handwriting.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.models import ResNet
from pyimagesearch.az_dataset import load_mnist_dataset
from pyimagesearch.az_dataset import load_az_dataset, show_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from PIL import Image


PREN_DATASET = "/root/ocr_pren_dataset/data_extracted"

def load_pren_dataset(datasetPath=PREN_DATASET):
    # initialize the list of data and labels
    data = []
    labels = []
    try:
        for direc in os.listdir(PREN_DATASET):
            for image_file in os.listdir(os.path.join(PREN_DATASET, direc)):
                img = cv2.imread(os.path.join(PREN_DATASET, direc, image_file), 0)
                # cv2.imshow("frame", img)
                # cv2.waitKey(0)
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                data.append(img)
                labels.append(direc)
    except Exception as e:
        print(str(e), image_file, direc)

    # images are represented as single channel (grayscale) images
    # that are 28x28=784 pixels -- we need to take this flattened
    # 784-d list of numbers and repshape them into a 28x28 matrix
    # image = image.reshape((28, 28))

    # # update the list of data and labels
    # data.append(image)
    # labels.append(label)

    # # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    # # return a 2-tuple of the A-Z data and labels
    return (data, labels)
    # print(type(data), type(labels), len(data), len(labels))



# load_pren_dataset(PREN_DATASET)
# print("[INFO] loading datasets...")
# (digitsData, digitsLabels) = load_mnist_dataset()
# print(type(digitsData), type(digitsLabels), len(digitsData), len(digitsLabels))
# # print(digitsData[0], digitsLabels[0])
# cv2.imshow("{}".format(digitsLabels[0]), digitsData[0])
# cv2.waitKey(0)

