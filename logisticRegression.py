from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os

from six.moves import urllib

import pandas as pd
import tensorflow as tf
from featureloader import featureloader

# load training features
train_data = featureloader('TRAIN', 'ECG5000')
df_train, feature_column = train_data.featureloader_UCR()
# df_train.to_csv('tmp_1.csv')

# load test training
test_data = featureloader('TEST', 'ECG5000')
df_test, feature_column = test_data.featureloader_UCR()
# df_test.to_csv('tmp_2.csv')

# remove \n in feature_column
feature_column[-1] = feature_column[-1].strip()

print(feature_column)
def input_fn(df, feature_column):
	feature_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in feature_column}
	label = tf.constant(df["label"].values)
	print(df["label"])
	return feature_cols, label

def train_input_fn():
	return input_fn(df_train, feature_column)

def eval_input_fn():
	return input_fn(df_test, feature_column)

# crossed_columns = tf.contrib.layers.crossed_columns(feature_column)
index = 0
layer=[]
for feature in feature_column:
	layer.append(tf.contrib.layers.real_valued_column(feature))
	index+= 1

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=layer, 
	model_dir=model_dir)
# m = tf.contrib.learn.DNNClassifier(feature_columns=layer, 
# 	model_dir=model_dir,
# 	hidden_units=[100,50])

m.fit(input_fn = train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
	print("%s: %s" % (key, results[key]))