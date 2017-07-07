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
from sklearn.decomposition import PCA 


class featureloader(object):
	"""docstring for featureloader"""
	data_type = ''
	data_name = ''
	def __init__(self, data_type, data_name):
		super(featureloader, self).__init__()
		self.data_type = data_type
		self.data_name = data_name

	def featureloader_UCR(self):
		data_file_name = 'result/' + self.data_name + '_features_' + self.data_type + '.csv'

		if not os.path.exists(data_file_name):
			raise RuntimeError('Feature file:'+ data_file_name +' not exist!')

		with open(data_file_name) as f:
			file_columns_line = f.readline().split(',')
			file_columns_line[-1] = file_columns_line[-1].strip()
			df = pd.read_csv(data_file_name, names = file_columns_line, skipinitialspace = True, skiprows = 1)
			if 'id' in file_columns_line:
				df.drop('id', axis = 1, inplace = True)
			# df.drop([df.columns[0]], axis = 1, inplace = True)
			column = df.columns.tolist()

		# Get label
		label_column = "label"
		label_file_name = os.path.join('/home/allen/Code/UCR_TS_Archive_2015/', self.data_name, self.data_name + '_' + self.data_type)
		if not os.path.exists(label_file_name):
			raise RuntimeError('Label file:' + label_file_name + ' not exist!')

		with open(label_file_name) as l:
			lf = pd.read_csv(label_file_name, header=None)

		print(len(lf[lf.columns[0]]))
		df[label_column] = lf[lf.columns[0]]

		return df, column

	def feature_PCA(self, df, num):
		data = df.as_matrix()
		pca = PCA(n_components=num)
		newData = pca.fit_transform(data)

		feature_column = [str(i) for i in range(num)]
		df = pd.DataFrame(data = newData, columns = feature_column)

		return df

