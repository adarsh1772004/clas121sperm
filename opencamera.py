import cv2
import numpy as np
from numpy.lib.index_tricks import MGridClass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import ssl
os.system('cls')
# Setting an HTTPS Context to fetchdata from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

# trainData=75%, testData=25%
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=7500, test_size=2500, random_state=9)


# scallingData
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0
