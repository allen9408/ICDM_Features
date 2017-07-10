from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os

from six.moves import urllib
import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf
from featureloader import featureloader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from result_figure import *
from sklearn import ensemble
# load training features
train_data = featureloader('TEST', 'ECG5000')
df_train, feature_column = train_data.featureloader_UCR()
# df_train.to_csv('tmp_1.csv')

# load test training
test_data = featureloader('TRAIN', 'ECG5000')
df_test, feature_column = test_data.featureloader_UCR()
# df_test.to_csv('tmp_2.csv')

# remove \n in feature_column
feature_column[-1] = feature_column[-1].strip()


# convert data to numpy
Y_train = df_train["label"].values
Y_test = df_test["label"].values
X_train = df_train.values
X_test = df_test.values

# scale input data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined_std = np.hstack((Y_train, Y_test))

clf = ensemble.AdaBoostClassifier()
clf.fit(X_train_std, Y_train)

# Get test result
Z = clf.predict(X_test_std)
with open("result/sklearn_reult.csv", "a") as o:
	o.write("Adabst, " + str(accuracy_score(Y_test, Z)) + "\n")

# plot_confusion_matrix(Y_test, Z, [0,1,2,3,4])
plot_result('ECG5000', Y_test, Z, clf, 'Adabst')
