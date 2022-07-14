import sys
import warnings

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from sklearn.datasets import load_iris

from os import listdir
from PIL import Image as PImage


onehot_encoder = OneHotEncoder(sparse=False)


import matplotlib.pyplot as plt

import seaborn as sns

# Postgresql
import psycopg2
from config import config
from psycopg2.extensions import register_adapter, AsIs

import cv2



class imdata:
    def __init__(self, id, im, imx, imy):
        self._id = id
        self._im = im
        self._imx = imx.flatten().astype(float) / 255.0
        self._imy = imy.flatten().astype(float) / 255.0
        self._dim = imx.shape

    def printx(self):
        cv2.imshow("x(" + str(self._id) + "-" + str(self._im) + ")", self._imx.reshape(self._dim))
        cv2.waitKey(0)

    def printy(self):
        cv2.imshow("y(" + str(self._id) + "-" + str(self._im) + ")", self._imy.reshape(self._dim))
        cv2.waitKey(0)





def loadImages(path, train):
    image_list = listdir(path)
    image_list = sorted(image_list)
    imx = []
    imy = []

    data = []
    files = []

    for image in image_list:
        if "mask" in image:
            continue
        else:
            files.append(image[:-4])

    for file in files:
        if train:
            id, im = file.split("_", 1)
            imx = cv2.imread(path + file + ".tif", cv2.IMREAD_GRAYSCALE)
            imy = cv2.imread(path + file + "_mask.tif", cv2.IMREAD_GRAYSCALE)
            data.append(imdata(id, im, imx, imy))
        else:
            id = file
            im = "0"
            imx = cv2.imread(path + file + ".tif", cv2.IMREAD_GRAYSCALE)
            imy = np.array([0])
            data.append(imdata(id, im, imx, imy))

    return data





def main(argv):
    train = loadImages("/home/icirauqui/w0rkspace_Arc/sn/datasets/ultrasound-nerve-segmentation/train/", True)
    test  = loadImages("/home/icirauqui/w0rkspace_Arc/sn/datasets/ultrasound-nerve-segmentation/test/", False)


    # Connect to Postgresql database
    params = config()
    conn = psycopg2.connect(**params)
    conn.autocommit = True
    cur = conn.cursor()

    # Create table
    query  = "DROP TABLE IF EXISTS nerves;"
    query += "CREATE TABLE IF NOT EXISTS nerves ("
    query += "  patient_id    INT,"
    query += "  image_id      INT,"
    query += "  imx           FLOAT ARRAY,"
    query += "  imy           FLOAT ARRAY"
    query += ")"
    cur.execute(query)

    # Create index
    cur.execute("CREATE INDEX patient_idx ON nerves(patient_id)")

    cur.execute("SET search_path = schema1, public;")

    # Load train set
    for i in range(len(train)):
        print("Inserting train into nerves:", i+1, "of", len(train))
        query  = "INSERT INTO nerves("
        #query += "id,"
        query += "patient_id,"
        query += "image_id  ,"
        query += "imx       ,"
        query += "imy       )"
        query += " VALUES ("
        query += str(train[i]._id) + ", "
        query += str(train[i]._im) + ", "
        query += "ARRAY" + str(train[i]._imx.tolist()) + ", "
        query += "ARRAY" + str(train[i]._imy.tolist())
        query += ")"
        cur.execute(query)

    # Load test set, patient id starting in 10000
    for i in range(len(test)):
        print("Inserting test into nerves:", i+1, "of", len(test))
        query  = "INSERT INTO nerves("
        #query += "id,"
        query += "patient_id,"
        query += "image_id  ,"
        query += "imx       "
        query += ") VALUES ("
        query += str(10000 + int(test[i]._id)) + ", "
        query += str(test[i]._im) + ", "
        query += "ARRAY" + str(test[i]._imx.tolist())
        query += ")"
        cur.execute(query)

    # Close connection
    cur.close()
    conn.close()


"""
    # Connect to Postgresql database
    params = config()
    conn = psycopg2.connect(**params)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT imx FROM nerves;")
    res1 = cur.fetchall()

    for r in res1:
        print(r)

    # Close connection
    cur.close()
    conn.close()
"""


if __name__ == '__main__':
    main(sys.argv)
